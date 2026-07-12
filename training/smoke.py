#!/usr/bin/env python3
"""Smoke + overfit test for MOSS-TTS-v1.5 (8B, 32-codebook delay). Proves the trainer design:
 encode audio -> 32-cb codes -> build computing_loss example -> mask prompt -> forward (finite loss)
 -> overfit 1 sample for N steps (loss must collapse) -> generate. If loss collapses, label masking is correct.

Usage: python smoke.py  (set MODEL_PATH to the base 8B checkpoint, WAV to any short speech clip)"""
import os, sys, glob, torch
M=os.environ.get("MODEL_PATH","./model"); dev="cuda:0"
from transformers import AutoProcessor, AutoModel
print("loading processor + codec...", flush=True)
proc=AutoProcessor.from_pretrained(M, trust_remote_code=True)
proc.audio_tokenizer=proc.audio_tokenizer.to(dev).eval()
print("loading 8B model...", flush=True)
model=AutoModel.from_pretrained(M, trust_remote_code=True, dtype=torch.bfloat16).to(dev)
NVQ=model.config.n_vq; print("n_vq =",NVQ, flush=True)
# any short speech clip to encode as the target
wav=os.environ.get("WAV") or (sorted(glob.glob("./refs/*.mp3")+glob.glob("./refs/*.wav")) or [None])[0]
assert wav, "set WAV=/path/to/clip.wav (or drop clips in ./refs/)"
print("test wav:",wav, flush=True)
codes=proc.encode_audios_from_path([wav], None)[0]
print("encoded codes shape:",tuple(codes.shape),"dtype",codes.dtype, flush=True)   # expect (T, 32)
INSTR="A warm, friendly voice, calm and clear. High quality studio recording."
TEXT="Hello there, this is a little test of the eight billion parameter model."
LANG="English"
Tframes=codes.shape[0] if codes.shape[0]<codes.shape[1] or codes.shape[1]==NVQ else codes.shape[1]
user=proc.build_user_message(text=TEXT, instruction=INSTR, language=LANG, tokens=int(Tframes), quality="high quality")
asst=proc.build_assistant_message(audio_codes_list=[codes])
full=proc([[user, asst]], mode="computing_loss")
promptonly=proc([[user]], mode="generation")
iid=full["input_ids"].to(dev); am=full["attention_mask"].to(dev)
P=promptonly["input_ids"].shape[1]
print("full seq:",tuple(iid.shape),"| prompt len:",P,"| attn sum:",int(am.sum()), flush=True)
labels=iid.clone()
labels[:, :P, :]=-100                       # mask the user prompt
labels[~am.bool()]=-100                      # mask any padding positions
# mask invalid targets: audio-head pad code (its logit is forced to -inf) + text pad
pad_audio=model.config.audio_pad_code; pad_txt=getattr(model.config,"pad_token_id",None) or model.config.language_config["pad_token_id"]
for c in range(1,NVQ+1):
    ch=labels[...,c]; ch[ch==pad_audio]=-100
labels[...,0][labels[...,0]==pad_txt]=-100
print("audio_pad_code",pad_audio,"pad_txt",pad_txt,"| supervised tokens:",int((labels!=-100).sum()),flush=True)
cw=[1.0]+[1.0]*NVQ
model.train()
out=model(input_ids=iid, attention_mask=am, labels=labels, channelwise_loss_weight=cw)
print("initial loss:", float(out.loss), flush=True)
assert torch.isfinite(out.loss), "loss not finite!"
# ---- mini overfit ----
opt=torch.optim.AdamW(model.parameters(), lr=1e-5)
for step in range(30):
    out=model(input_ids=iid, attention_mask=am, labels=labels, channelwise_loss_weight=cw)
    opt.zero_grad(); out.loss.backward(); opt.step()
    if step%5==0: print(f"  overfit step {step}: loss {float(out.loss):.4f}", flush=True)
print("FINAL overfit loss:", float(out.loss), flush=True)
print("SMOKE OK" if float(out.loss) < float(0.9)* 1 else "CHECK: loss did not drop much", flush=True)
