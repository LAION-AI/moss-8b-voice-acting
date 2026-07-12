#!/usr/bin/env python3
"""Generate a handful of held-out emotional samples from a trained MOSS-8B checkpoint, decode to wav,
and write a manifest (handy for a live training dashboard). Fixes the checkpoint's copied support .py
files (the trainer may copy a variant set) before loading. No reference audio."""
import os, sys, json, shutil, argparse, glob, torch, torchaudio
os.environ.setdefault("HF_HOME","/tmp/hf_cache")
# ISOLATE the dynamic-module cache for generation so we NEVER touch the trainer's cache.
# Deleting the trainer's transformers_modules cache would crash its checkpoint save
# (save_pretrained copies modeling_moss_tts.py from that cache).
os.environ["HF_MODULES_CACHE"]=os.environ.get("GEN_MODULES_CACHE","./gen_modules_cache")
MODEL_DIR=os.environ.get("MODEL_PATH","./model")  # base ckpt holding the canonical support .py files
from transformers import AutoProcessor, AutoModel

PROMPTS=[
 ("Heartbroken & tearful", "A heartbroken woman, voice trembling and thick with tears, barely holding back sobs as she speaks.",
  "I keep reaching for the phone to call you... and then I remember you're never going to pick up again."),
 ("Ecstatic joy", "An ecstatic young man overflowing with pure joy, laughing between words, breathless with excitement.",
  "We did it! We actually did it! I can't believe it, this is the best day of my entire life!"),
 ("Cold fury / rage", "A man speaking with cold, controlled fury, every word clipped and seething with barely contained rage.",
  "You think I don't know what you did? Get out. Get out before I do something we'll both regret."),
 ("Gentle whispered tenderness", "A mother gently whispering with soft, warm tenderness, soothing a sleepy child.",
  "Shhh, my love, close your eyes now. I'm right here, and I will never, ever let anything hurt you."),
 ("Terrified panic", "A terrified person gasping in panic, voice shaking and rushed, on the edge of screaming.",
  "No no no, it's right behind us, don't look back, just run, run, we have to get out of here now!"),
 ("Warm playful humor", "A warm, playful storyteller with a mischievous grin in the voice, teasing and full of good humor.",
  "Oh, so NOW you want my advice? After you spectacularly ignored it and set the kitchen on fire? Classic."),
]

def fix_support_files(ckpt):
    for f in ["configuration_moss_tts.py","modeling_moss_tts.py","processing_moss_tts.py",
              "inference_utils.py","tts_robust_normalizer_single_script.py","__init__.py"]:
        src=os.path.join(MODEL_DIR,f); dst=os.path.join(ckpt,f)
        if os.path.exists(src) and os.path.abspath(src)!=os.path.abspath(dst):
            shutil.copy(src, dst)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--tag",default="")
    ap.add_argument("--outdir",default="./dash")
    a=ap.parse_args()
    os.makedirs(a.outdir,exist_ok=True)
    fix_support_files(a.ckpt)
    # purge ONLY the isolated gen cache (never the trainer's modules cache)
    for d in glob.glob(os.path.join(os.environ["HF_MODULES_CACHE"],"transformers_modules","*")):
        try: shutil.rmtree(d)
        except Exception: pass
    dev="cuda:0"
    try: proc=AutoProcessor.from_pretrained(a.ckpt, trust_remote_code=True)
    except Exception: proc=AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    proc.audio_tokenizer=proc.audio_tokenizer.to(dev).eval()
    model=AutoModel.from_pretrained(a.ckpt, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    sr=getattr(getattr(proc.audio_tokenizer,'config',None),'sampling_rate',None) or int(proc.model_config.sampling_rate)
    manifest=[]
    for i,(label,inst,text) in enumerate(PROMPTS,1):
        best=None
        for s in range(2):
            torch.manual_seed(s)
            conv=[[proc.build_user_message(text=text, instruction=inst, language="English")]]
            batch=proc(conv, mode="generation")
            with torch.no_grad():
                out=model.generate(input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev),
                    max_new_tokens=1200, text_temperature=0.7, audio_temperature=0.9, audio_top_p=0.95,
                    audio_top_k=25, audio_repetition_penalty=1.1)
            msg=proc.decode(out)[0]
            if not msg.audio_codes_list: continue
            wav=msg.audio_codes_list[0].cpu().float()
            if wav.dim()==1: wav=wav.unsqueeze(0)
            dur=wav.shape[-1]/sr; rms=float(wav.pow(2).mean().sqrt())
            if rms>0.005 and (best is None or dur>best[0]): best=(dur,wav)
        fn=f"sample{i}.wav"
        if best is not None:
            torchaudio.save(os.path.join(a.outdir,fn), best[1], sr)
            manifest.append({"file":fn,"label":label,"text":text,"instruction":inst,"dur":round(best[0],1)})
            print(f"[{i}] {label}: dur={best[0]:.1f}s OK", flush=True)
        else:
            print(f"[{i}] {label}: FAILED (silent)", flush=True)
    outj={"tag":a.tag,"samples":manifest}
    tmp=os.path.join(a.outdir,"samples.json.tmp"); json.dump(outj,open(tmp,"w")); os.replace(tmp,os.path.join(a.outdir,"samples.json"))
    print(f"SAMPLES_DONE tag={a.tag} n={len(manifest)}", flush=True)

if __name__=="__main__": main()
