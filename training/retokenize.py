#!/usr/bin/env python3
"""Re-tokenize a local WebDataset (tar shards) of speech samples to 32-codebook MOSS-8B codes
and emit trainer-ready JSONL. Each output record:
  {id, text, instruction, language, tokens, quality, audio_codes, [reference_audio_codes], bucket}
Run one source per process, pinned to one GPU via CUDA_VISIBLE_DEVICES.

Data model: your corpus is a directory of tar shards; each sample is a group of files sharing a
stem, e.g.  <stem>.json (metadata with the text + instruction), <stem>.audio.mp3 (target speech),
and optionally <stem>.ref.mp3 (a different clip of the same speaker, for voice-cloning pairs).
Point --data at that directory and set --text-keys / --instr-keys to your JSON field names."""
import os, sys, io, json, glob, tarfile, tempfile, argparse, traceback
import torch, torchaudio, langid
langid.set_languages(['en','de','zh','fr','es','it','pt','ru','ja','ko','nl','fi','sv','el','he','pl','tr','ar','hi','vi','th'])
ISO2NAME={'en':'English','de':'German','zh':'Chinese','yue':'Cantonese','fr':'French','es':'Spanish',
 'it':'Italian','pt':'Portuguese','ru':'Russian','ja':'Japanese','ko':'Korean','nl':'Dutch','fi':'Finnish',
 'sv':'Swedish','el':'Greek','he':'Hebrew','pl':'Polish','tr':'Turkish','ar':'Arabic','hi':'Hindi',
 'vi':'Vietnamese','th':'Thai'}
M=os.environ.get("MODEL_PATH","./model")

def pick(d, keys):
    for k in keys:
        v=d.get(k)
        if v and str(v).strip(): return str(v).strip()
    return None

def detect_lang(text, default):
    if default is not None: return default
    try:
        code,_=langid.classify(text[:400]); return ISO2NAME.get(code,"English")
    except Exception: return "English"

def group_tar(tarpath):
    """Yield (stem, {suffix:bytes}) per sample from a webdataset-style tar."""
    with tarfile.open(tarpath,"r") as tf:
        by={}
        for m in tf.getmembers():
            if not m.isfile(): continue
            stem=m.name.split('.',1)[0]; suf=m.name.split('.',1)[1] if '.' in m.name else ""
            by.setdefault(stem,{})[suf]=m
        for stem,d in by.items():
            out={}
            for suf,m in d.items():
                if suf.endswith(("json",)) or suf.endswith(("audio.mp3","ref.mp3")):
                    out[suf]=tf.extractfile(m).read()
            yield stem,out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True,help="glob for tar shards, e.g. '/path/<YOUR_DATASET>/*.tar'")
    ap.add_argument("--out",required=True)
    ap.add_argument("--text-keys",default="text,raw_text,transcription,asr_transcript",help="comma list of JSON field candidates for the spoken text")
    ap.add_argument("--instr-keys",default="instruction,prompt,prompt_full",help="comma list of JSON field candidates for the style instruction")
    ap.add_argument("--language",default=None,help="fixed language name, or omit to auto-detect")
    ap.add_argument("--clone-mode",default="none",choices=["none","split","pair"],
                    help="split: alternate samples become voice-clone pairs using <stem>.ref.mp3; pair: always clone if ref present")
    ap.add_argument("--bucket",default="data")
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--batch",type=int,default=12)
    args=ap.parse_args()
    text_keys=args.text_keys.split(","); instr_keys=args.instr_keys.split(",")

    from transformers import AutoProcessor
    dev="cuda:0"
    proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
    proc.audio_tokenizer=proc.audio_tokenizer.to(dev).eval()
    tmpd=tempfile.mkdtemp(dir=os.environ.get("TMPDIR","/tmp"),prefix="enc_")
    counts={}
    fout=open(args.out,"w")

    def encode_batch(items):
        paths=[it["tgt_path"] for it in items]
        with torch.no_grad():
            codes=proc.encode_audios_from_path(paths,None)
        refpaths=[it.get("ref_path") for it in items]
        refcodes=[None]*len(items)
        todo=[(i,p) for i,p in enumerate(refpaths) if p]
        if todo:
            with torch.no_grad():
                rc=proc.encode_audios_from_path([p for _,p in todo],None)
            for (i,_),c in zip(todo,rc): refcodes[i]=c
        return codes,refcodes

    def flush(items):
        if not items: return 0
        n=0
        try:
            codes,refcodes=encode_batch(items)
        except Exception:
            codes=[]; refcodes=[]
            for it in items:
                try:
                    with torch.no_grad(): c=proc.encode_audios_from_path([it["tgt_path"]],None)[0]
                    rc=None
                    if it.get("ref_path"):
                        with torch.no_grad(): rc=proc.encode_audios_from_path([it["ref_path"]],None)[0]
                    codes.append(c); refcodes.append(rc)
                except Exception:
                    codes.append(None); refcodes.append(None)
        for it,c,rc in zip(items,codes,refcodes):
            if c is None or c.shape[0]<4: continue
            rec={"id":it["id"],"text":it["text"],"instruction":it["instruction"],
                 "language":it["language"],"tokens":int(c.shape[0]),"quality":"high quality",
                 "audio_codes":c.to(torch.int32).tolist(),"bucket":it["bucket"]}
            if it.get("clone") and rc is not None and rc.shape[0]>=4:
                rec["reference_audio_codes"]=[rc.to(torch.int32).tolist()]
                rec["bucket"]=it["bucket"]+"_clone"
            fout.write(json.dumps(rec,ensure_ascii=False)+"\n")
            counts[rec["bucket"]]=counts.get(rec["bucket"],0)+1
            n+=1
        return n

    tars=sorted(glob.glob(args.data))
    got=0; buf=[]; idx=0
    for tp in tars:
        for stem,bag in group_tar(tp):
            if args.limit and got>=args.limit: break
            jb=bag.get("json")
            if jb is None: continue
            audio=bag.get("audio.mp3")
            if audio is None: continue
            try: meta=json.loads(jb)
            except Exception: continue
            text=pick(meta,text_keys); instr=pick(meta,instr_keys)
            if not text or not instr or len(text)<2: continue
            lang=detect_lang(text,args.language)
            tpath=os.path.join(tmpd,f"{args.bucket}_{idx}_t.mp3"); open(tpath,"wb").write(audio)
            clone=False; rpath=None
            if args.clone_mode!="none" and bag.get("ref.mp3"):
                clone = (idx%2==0) if args.clone_mode=="split" else True
                if clone:
                    rpath=os.path.join(tmpd,f"{args.bucket}_{idx}_r.mp3"); open(rpath,"wb").write(bag["ref.mp3"])
            buf.append({"id":f"{args.bucket}__{stem}","text":text,"instruction":instr,"language":lang,
                        "tgt_path":tpath,"ref_path":rpath,"clone":clone,"bucket":args.bucket})
            idx+=1; got+=1
            if len(buf)>=args.batch:
                flush(buf)
                for it in buf:
                    for p in (it["tgt_path"],it.get("ref_path")):
                        if p and os.path.exists(p): os.remove(p)
                buf=[]; fout.flush()
        if args.limit and got>=args.limit: break
    flush(buf)
    for it in buf:
        for p in (it["tgt_path"],it.get("ref_path")):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass
    fout.close()
    print("COUNTS "+json.dumps(counts),flush=True)

if __name__=="__main__":
    main()
