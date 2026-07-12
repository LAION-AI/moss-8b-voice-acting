#!/usr/bin/env python3
"""Sweep batch size and measure MOSS-TTS-v1.5 8B generation throughput.
Reports, per batch B: wall time, clips/s, total audio seconds, RTF (audio-s / wall-s),
generated tokens, tokens/s, and peak VRAM. Writes throughput.json.

    python throughput_benchmark.py --model /path/to/checkpoint
"""
import os, sys, time, json, argparse, torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=os.environ.get("MOSS_MODEL","laion/moss-tts-v1.5-8b-voice-acting"))
    ap.add_argument("--batches",default="1,2,4,8,16,24,32,48,64")
    ap.add_argument("--max-new-tokens",type=int,default=200,help="~10s target (model ~18 audio frames/s)")
    ap.add_argument("--out",default="throughput.json")
    a=ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    dev="cuda"
    from transformers import AutoModel, AutoProcessor
    proc=AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    proc.audio_tokenizer=proc.audio_tokenizer.to(dev)
    model=AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                    attn_implementation="sdpa").to(dev).eval()
    SR=proc.model_config.sampling_rate
    TXT=("We stand at the edge of a new era, and the road ahead is long and bright. "
         "Every step we take together carries the weight of hope and the promise of tomorrow, "
         "so let us walk on with courage, with patience, and with unshakable resolve.")

    def run_batch(B):
        um=proc.build_user_message(text=TXT, instruction="A clear neutral narrator voice. High quality recording.",
                                   language="English", quality="high quality")
        conv=[[um] for _ in range(B)]
        b=proc(conv, mode="generation")
        ii=b["input_ids"].to(dev); am=b["attention_mask"].to(dev)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0=time.time()
        with torch.no_grad():
            out=model.generate(input_ids=ii, attention_mask=am, max_new_tokens=a.max_new_tokens,
                audio_temperature=1.0, audio_top_p=0.95, audio_top_k=25, audio_repetition_penalty=1.1)
        torch.cuda.synchronize(); wall=time.time()-t0
        peak=torch.cuda.max_memory_allocated()/1e9
        msgs=proc.decode(out)
        auds=[m.audio_codes_list[0] for m in msgs if m.audio_codes_list]
        tot_audio=sum(x.shape[-1]/SR for x in auds); n=len(auds)
        steps=out.shape[-1]-ii.shape[-1] if hasattr(out,"shape") else a.max_new_tokens
        return dict(B=B, wall=round(wall,2), n_ok=n, peak_gb=round(peak,1),
            samples_per_s=round(B/wall,3), audio_s=round(tot_audio,1),
            rtf=round(tot_audio/wall,3), gen_tokens=int(steps),
            tokens_per_s=round(B*steps/wall,1), mean_audio_s=round(tot_audio/max(n,1),2))

    res=[]
    try: run_batch(1)   # warmup
    except Exception as e: print("warmup err",e,flush=True)
    for B in [int(x) for x in a.batches.split(",")]:
        try:
            r=run_batch(B); res.append(r); print(json.dumps(r),flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at B={B}",flush=True); torch.cuda.empty_cache(); break
        except Exception as e:
            print(f"ERR at B={B}: {e}",flush=True); torch.cuda.empty_cache(); break
    json.dump(res, open(a.out,"w"))
    print("THROUGHPUT DONE",flush=True)

if __name__=="__main__": main()
