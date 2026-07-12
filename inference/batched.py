#!/usr/bin/env python3
"""FAST path: batched generation with MOSS-TTS-v1.5 8B.

Single-stream decoding is architecturally fixed at ~32 codebook-steps/s (~2x realtime), so the
lever for throughput is BATCHING. Two common patterns:

  1) best-of-N sampling of ONE prompt: replicate the prompt B times, one generate() call returns
     B independent takes; pick the best with your scorer. (Shown here.)
  2) many DIFFERENT prompts at once: pad a list of prompts into one batch (same call).

bf16 + attn_implementation="sdpa" is the portable fast default. If `flash_attn` is installed,
attn_implementation="flash_attention_2" is faster still. See FAST_INFERENCE.md for the measured
throughput curve (B=1..64) and scaling analysis.

    python batched.py --text "Hello, world." --n 16 --outdir takes/
"""
import os, argparse, torch, torchaudio

MODEL=os.environ.get("MOSS_MODEL","laion/moss-tts-v1.5-8b-voice-acting")

def pick_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=MODEL)
    ap.add_argument("--text",required=True)
    ap.add_argument("--instruction",default="A clear, natural voice. High quality recording.")
    ap.add_argument("--language",default="English")
    ap.add_argument("--n",type=int,default=16,help="batch size = number of takes (best-of-N)")
    ap.add_argument("--reference",default=None)
    ap.add_argument("--outdir",default="takes")
    ap.add_argument("--max-new-tokens",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    os.makedirs(a.outdir,exist_ok=True)

    dev="cuda" if torch.cuda.is_available() else "cpu"
    attn=pick_attn()
    from transformers import AutoModel, AutoProcessor
    proc=AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    proc.audio_tokenizer=proc.audio_tokenizer.to(dev).eval()
    model=AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                    attn_implementation=attn).to(dev).eval()
    sr=getattr(getattr(proc.audio_tokenizer,'config',None),'sampling_rate',None) or int(proc.model_config.sampling_rate)
    print(f"loaded {a.model}  attn={attn}  sr={sr}", flush=True)

    kw=dict(text=a.text, instruction=a.instruction, language=a.language, quality="high quality")
    if a.reference:
        ref=proc.encode_audios_from_path([a.reference], None)
        kw["reference"]=[ref[0]]; kw["tokens"]=int(ref[0].shape[0])
        audio_temp=1.0
    else:
        audio_temp=0.8
    um=proc.build_user_message(**kw)

    # Replicate the SAME prompt B=n times -> one generate() call returns n independent takes.
    conv=[[um] for _ in range(a.n)]
    batch=proc(conv, mode="generation")

    torch.manual_seed(a.seed)
    with torch.no_grad():
        out=model.generate(input_ids=batch["input_ids"].to(dev),
            attention_mask=batch["attention_mask"].to(dev),
            max_new_tokens=a.max_new_tokens, text_temperature=0.7,
            audio_temperature=audio_temp, audio_top_p=0.95, audio_top_k=25,
            audio_repetition_penalty=1.1)
    msgs=proc.decode(out)

    n_ok=0
    for i,msg in enumerate(msgs):
        if not msg.audio_codes_list: continue
        wav=msg.audio_codes_list[0].cpu().float()
        if wav.dim()==1: wav=wav.unsqueeze(0)
        if float(wav.abs().max())<0.005: continue   # skip silent takes
        torchaudio.save(os.path.join(a.outdir,f"take{i:02d}.wav"), wav, sr)
        n_ok+=1
    print(f"wrote {n_ok}/{a.n} takes to {a.outdir}/  "
          f"(now score them with your quality/genuineness model and keep the best)")

if __name__=="__main__": main()
