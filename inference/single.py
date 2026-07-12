#!/usr/bin/env python3
"""Minimal single-utterance generation with MOSS-TTS-v1.5 8B voice-acting.

Loads the model + processor from a local checkpoint dir OR the HF model repo, builds one
user message (style instruction + text, optional reference clip for voice cloning) and runs
one model.generate, then writes a 24 kHz wav.

    python single.py --text "Hello, world." \
        --instruction "A warm, friendly narrator. High quality studio recording." \
        --out out.wav
    # voice cloning: add  --reference /path/to/speaker.wav
"""
import os, argparse, torch, torchaudio

MODEL=os.environ.get("MOSS_MODEL","laion/moss-tts-v1.5-8b-voice-acting")  # or a local ckpt dir

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=MODEL,help="local checkpoint dir or HF repo id")
    ap.add_argument("--text",required=True)
    ap.add_argument("--instruction",default="A clear, natural voice. High quality recording.")
    ap.add_argument("--language",default="English")
    ap.add_argument("--reference",default=None,help="optional reference wav/mp3 for voice cloning")
    ap.add_argument("--out",default="out.wav")
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()

    dev="cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoProcessor
    proc=AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    proc.audio_tokenizer=proc.audio_tokenizer.to(dev).eval()
    model=AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16,
                                    attn_implementation="eager").to(dev).eval()
    sr=getattr(getattr(proc.audio_tokenizer,'config',None),'sampling_rate',None) or int(proc.model_config.sampling_rate)

    kw=dict(text=a.text, instruction=a.instruction, language=a.language, quality="high quality")
    # Recommended sampling settings from our grid search:
    #   with a reference clip  -> audio_temperature 1.0, rep-pen 1.1
    #   without a reference     -> audio_temperature 0.8, rep-pen 1.1
    if a.reference:
        ref=proc.encode_audios_from_path([a.reference], None)  # list of (T, 32)
        kw["reference"]=[ref[0]]; kw["tokens"]=int(ref[0].shape[0])
        audio_temp=1.0
    else:
        audio_temp=0.8
    um=proc.build_user_message(**kw)
    batch=proc([[um]], mode="generation")

    torch.manual_seed(a.seed)
    with torch.no_grad():
        out=model.generate(input_ids=batch["input_ids"].to(dev),
            attention_mask=batch["attention_mask"].to(dev),
            max_new_tokens=1200, text_temperature=0.7,
            audio_temperature=audio_temp, audio_top_p=0.95, audio_top_k=25,
            audio_repetition_penalty=1.1)
    msg=proc.decode(out)[0]
    assert msg.audio_codes_list, "generation produced no audio"
    wav=msg.audio_codes_list[0].cpu().float()
    if wav.dim()==1: wav=wav.unsqueeze(0)
    torchaudio.save(a.out, wav, sr)
    print(f"wrote {a.out}  ({wav.shape[-1]/sr:.1f}s @ {sr} Hz)")

if __name__=="__main__": main()
