#!/usr/bin/env python3
"""Stream a remote WebDataset (HF Hub tar shards), re-tokenize audio to 32-cb MOSS-8B codes,
emit trainer JSONL. One source + a slice of tars per process, pinned to one GPU.

Set --source to your own HF dataset repo id (or any repo you have access to) and adjust the
field names / audio extension to your schema. Nothing here is specific to any particular corpus."""
import os, sys, io, json, tarfile, tempfile, argparse, requests
import torch, langid
langid.set_languages(['en','de','zh','fr','es','it','pt','ru','ja','ko','nl','fi','sv','el','he','pl','tr','ar','hi','vi','th'])
ISO2NAME={'en':'English','de':'German','zh':'Chinese','yue':'Cantonese','fr':'French','es':'Spanish',
 'it':'Italian','pt':'Portuguese','ru':'Russian','ja':'Japanese','ko':'Korean','nl':'Dutch','fi':'Finnish',
 'sv':'Swedish','el':'Greek','he':'Hebrew','pl':'Polish','tr':'Turkish','ar':'Arabic','hi':'Hindi',
 'vi':'Vietnamese','th':'Thai'}
TOK=os.environ.get("HF_TOKEN")
M=os.environ.get("MODEL_PATH","./model")
from huggingface_hub import hf_hub_url, list_repo_files

def pick(d,keys):
    if not keys: return None
    for k in keys:
        v=d.get(k)
        if v and str(v).strip(): return str(v).strip()
    return None
def detect_lang(text,mode,meta):
    if mode=="English": return "English"
    if mode=="json":
        lv=meta.get("language") or meta.get("language_id")
        if lv:
            lv=str(lv).strip()
            if lv in ISO2NAME: return ISO2NAME[lv]
            if lv in ISO2NAME.values(): return lv
    try: return ISO2NAME.get(langid.classify(text[:400])[0],"English")
    except Exception: return "English"

def stream_tar(repo, tarfn):
    url=hf_hub_url(repo,tarfn,repo_type="dataset")
    r=requests.get(url,headers={"Authorization":f"Bearer {TOK}"},stream=True,timeout=300); r.raw.decode_content=True
    mode="r|gz" if tarfn.endswith(".gz") else "r|"
    tf=tarfile.open(fileobj=r.raw,mode=mode)
    bag={}; curstem=None
    for m in tf:
        if not m.isfile(): continue
        stem=m.name.split('.',1)[0]; suf=m.name.split('.',1)[1] if '.' in m.name else ""
        if curstem is None: curstem=stem
        if stem!=curstem and bag:
            yield curstem,bag; bag={}; curstem=stem
        if suf.endswith("json") or suf in ("mp3","flac","wav","ref.mp3","ogg"):
            bag[suf]=tf.extractfile(m).read()
    if bag: yield curstem,bag
    r.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,help="HF dataset repo id (or <YOUR_DATASET>)")
    ap.add_argument("--out",required=True)
    ap.add_argument("--tars",required=True,help="comma list of tar filenames, or 'ALL'")
    ap.add_argument("--audio-ext",default="mp3",help="primary audio extension in each sample")
    ap.add_argument("--ref-ext",default="",help="reference-clip extension for voice-clone pairs, or empty")
    ap.add_argument("--text-keys",default="text,transcription,asr_transcript")
    ap.add_argument("--instr-keys",default="instruction,prompt,prompt_full")
    ap.add_argument("--language",default="json",help="fixed name, 'json' (read from metadata), or 'auto'")
    ap.add_argument("--clone-mode",default="none",choices=["none","split"])
    ap.add_argument("--limit",type=int,default=0); ap.add_argument("--batch",type=int,default=12)
    a=ap.parse_args()
    text_keys=a.text_keys.split(","); instr_keys=a.instr_keys.split(",")
    from transformers import AutoProcessor
    dev="cuda:0"
    proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
    proc.audio_tokenizer=proc.audio_tokenizer.to(dev).eval()
    tmpd=tempfile.mkdtemp(dir=os.environ.get("TMPDIR","/tmp"),prefix="hfenc_")
    if a.tars=="ALL":
        allf=list_repo_files(a.source,repo_type="dataset",token=TOK)
        tars=[f for f in allf if f.endswith((".tar",".tar.gz"))]
    else:
        tars=a.tars.split(",")
    fout=open(a.out,"w"); counts={}; got=0

    def flush(items):
        nonlocal got
        if not items: return
        paths=[it["tp"] for it in items]
        try:
            with torch.no_grad(): codes=proc.encode_audios_from_path(paths,None)
        except Exception:
            codes=[]
            for it in items:
                try:
                    with torch.no_grad(): codes.append(proc.encode_audios_from_path([it["tp"]],None)[0])
                except Exception: codes.append(None)
        refc=[None]*len(items)
        todo=[(i,it["rp"]) for i,it in enumerate(items) if it.get("rp")]
        if todo:
            try:
                with torch.no_grad(): rr=proc.encode_audios_from_path([p for _,p in todo],None)
                for (i,_),c in zip(todo,rr): refc[i]=c
            except Exception: pass
        for it,c,rc in zip(items,codes,refc):
            if c is None or c.shape[0]<4: continue
            rec={"id":it["id"],"text":it["text"],"instruction":it["instruction"],"language":it["lang"],
                 "tokens":int(c.shape[0]),"quality":"high quality","audio_codes":c.to(torch.int32).tolist(),
                 "bucket":it["bucket"]}
            if it.get("clone") and rc is not None and rc.shape[0]>=4:
                rec["reference_audio_codes"]=[rc.to(torch.int32).tolist()]; rec["bucket"]=it["bucket"]+"_clone"
            fout.write(json.dumps(rec,ensure_ascii=False)+"\n")
            counts[rec["bucket"]]=counts.get(rec["bucket"],0)+1; got+=1
        for it in items:
            for p in (it["tp"],it.get("rp")):
                if p and os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass

    bucket=os.path.basename(a.source).replace("/","_")
    buf=[]; idx=0
    for ti,tarfn in enumerate(tars):
        if a.limit and got>=a.limit: break
        try:
            for stem,bag in stream_tar(a.source,tarfn):
                if a.limit and got+len(buf)>=a.limit: break
                jb=bag.get("json") or bag.get("ref.json")
                if jb is None: continue
                try: meta=json.loads(jb)
                except Exception: continue
                aud=None; ae=None
                for e in (a.audio_ext,"mp3","flac","wav","ogg"):
                    if e and bag.get(e): aud=bag[e]; ae=e; break
                if aud is None: continue
                text=pick(meta,text_keys); instr=pick(meta,instr_keys)
                if not text or not instr or len(text)<2: continue
                lang=detect_lang(text,a.language,meta)
                tp=os.path.join(tmpd,f"{bucket}_{idx}_t.{ae}"); open(tp,"wb").write(aud)
                clone=False; rp=None
                if a.clone_mode=="split" and a.ref_ext and bag.get(a.ref_ext):
                    clone=(idx%2==0)
                    if clone:
                        rp=os.path.join(tmpd,f"{bucket}_{idx}_r.mp3"); open(rp,"wb").write(bag[a.ref_ext])
                buf.append({"id":f"{bucket}__{stem}","text":text,"instruction":instr,"lang":lang,
                            "tp":tp,"rp":rp,"clone":clone,"bucket":bucket})
                idx+=1
                if len(buf)>=a.batch: flush(buf); buf=[]; fout.flush()
        except Exception as e:
            print(f"[warn] tar {tarfn}: {e}",flush=True); continue
        print(f"[{bucket}] {ti+1}/{len(tars)} tars, got={got}",flush=True)
    flush(buf); fout.close()
    print("COUNTS "+json.dumps(counts),flush=True)

if __name__=="__main__": main()
