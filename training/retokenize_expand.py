#!/usr/bin/env python3
"""Re-tokenize a PAIRED WebDataset (each sample has a target clip + a reference clip of the same
speaker) to 32-cb MOSS-8B codes, emitting SEVERAL training records per source group to maximize data:
  - a standalone record (target audio, full style instruction)
  - a voice-clone record (target audio + reference codes, focused instruction)
  - optionally a second standalone record from the reference clip
Stream from an HF dataset repo (or adapt stream/glob for local tars). One source + tar slice per GPU.

Expected per-sample files in each tar (rename via the --*-suffix flags to match your schema):
  <base>_tgt.mp3   target speech
  <base>_ref.mp3   a different clip of the same speaker (optional)
  <base>.json      metadata: full instruction, focused instruction, transcript(s), language
"""
import os, io, json, tarfile, tempfile, argparse, requests
import torch, langid
langid.set_languages(['en','de','zh','fr','es','it','pt','ru','ja','ko','nl','fi','sv','el','he','pl','tr','ar','hi','vi','th'])
ISO2NAME={'en':'English','de':'German','zh':'Chinese','yue':'Cantonese','fr':'French','es':'Spanish',
 'it':'Italian','pt':'Portuguese','ru':'Russian','ja':'Japanese','ko':'Korean','nl':'Dutch','fi':'Finnish',
 'sv':'Swedish','el':'Greek','he':'Hebrew','pl':'Polish','tr':'Turkish','ar':'Arabic','hi':'Hindi',
 'vi':'Vietnamese','th':'Thai'}
TOK=os.environ.get("HF_TOKEN")
M=os.environ.get("MODEL_PATH","./model")
from huggingface_hub import hf_hub_url, list_repo_files

def lang_of(meta):
    lv=meta.get("language") or meta.get("language_id") or ""
    lv=str(lv).strip()
    if lv in ISO2NAME: return ISO2NAME[lv]
    if lv in ISO2NAME.values(): return lv
    t=meta.get("text") or meta.get("asr_transcript") or ""
    try: return ISO2NAME.get(langid.classify(t[:400])[0],"English")
    except Exception: return "English"

def read_tar_groups(repo, tarfn, tgt_sfx, ref_sfx):
    """Return dict base_key -> {'tgt':bytes,'ref':bytes,'json':meta}."""
    url=hf_hub_url(repo,tarfn,repo_type="dataset")
    r=requests.get(url,headers={"Authorization":f"Bearer {TOK}"},stream=True,timeout=600); r.raw.decode_content=True
    mode_open="r|gz" if tarfn.endswith(".gz") else "r|"
    tf=tarfile.open(fileobj=r.raw,mode=mode_open)
    groups={}
    for m in tf:
        if not m.isfile(): continue
        nm=m.name; base=None; role=None
        if nm.endswith(tgt_sfx): base=nm[:-len(tgt_sfx)]; role="tgt"
        elif ref_sfx and nm.endswith(ref_sfx): base=nm[:-len(ref_sfx)]; role="ref"
        elif nm.endswith(".json"): base=nm[:-5]; role="json"
        else: continue
        g=groups.setdefault(base,{})
        if role=="json":
            try: g["json"]=json.loads(tf.extractfile(m).read())
            except Exception: pass
        else:
            g[role]=tf.extractfile(m).read()
    r.close()
    return groups

def build_specs(base, g, bucket, full_keys, focus_keys, text_keys):
    """Yield record specs (standalone + clone) for one source group."""
    meta=g.get("json")
    if not meta: return []
    def pick(keys):
        for k in keys:
            v=meta.get(k)
            if v and str(v).strip(): return str(v).strip()
        return None
    lang=lang_of(meta); out=[]
    full=pick(full_keys); focus=pick(focus_keys); text=pick(text_keys)
    if g.get("tgt") and full and text:
        out.append(dict(audio="tgt",ref_from=None,text=text,instr=full,lang=lang,bucket=bucket,idbase=base+"_std"))
    if g.get("tgt") and g.get("ref") and focus and text:
        out.append(dict(audio="tgt",ref_from="ref",text=text,instr=focus,lang=lang,bucket=bucket+"_clone",idbase=base+"_cl"))
    return [s for s in out if s["text"] and s["instr"] and len(str(s["text"]))>=2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,help="HF dataset repo id (or <YOUR_DATASET>)")
    ap.add_argument("--out",required=True)
    ap.add_argument("--tar-start",type=int,default=0); ap.add_argument("--tar-end",type=int,default=10**9)
    ap.add_argument("--tgt-suffix",default="_tgt.mp3"); ap.add_argument("--ref-suffix",default="_ref.mp3")
    ap.add_argument("--full-keys",default="prompt_full,instruction"); ap.add_argument("--focus-keys",default="prompt_focused,prompt")
    ap.add_argument("--text-keys",default="text,asr_transcript,transcription")
    ap.add_argument("--batch",type=int,default=16)
    a=ap.parse_args()
    bucket=os.path.basename(a.source).replace("/","_")
    full_keys=a.full_keys.split(","); focus_keys=a.focus_keys.split(","); text_keys=a.text_keys.split(",")
    from transformers import AutoProcessor
    dev="cuda:0"
    proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
    proc.audio_tokenizer=proc.audio_tokenizer.to(dev).eval()
    tmpd=tempfile.mkdtemp(dir=os.environ.get("TMPDIR","/tmp"),prefix="exp_")
    allf=list_repo_files(a.source,repo_type="dataset",token=TOK)
    tars=sorted(f for f in allf if f.endswith((".tar",".tar.gz")))
    tars=tars[a.tar_start:a.tar_end]
    fout=open(a.out,"w"); counts={}

    def encode_paths(paths):
        try:
            with torch.no_grad(): return proc.encode_audios_from_path(paths,None)
        except Exception:
            res=[]
            for p in paths:
                try:
                    with torch.no_grad(): res.append(proc.encode_audios_from_path([p],None)[0])
                except Exception: res.append(None)
            return res

    for ti,tarfn in enumerate(tars):
        try:
            groups=read_tar_groups(a.source,tarfn,a.tgt_suffix,a.ref_suffix)
        except Exception as e:
            print(f"[warn] tar {tarfn}: {e}",flush=True); continue
        idx=0; path_of={}
        for base,g in groups.items():
            for role in ("tgt","ref"):
                if g.get(role) and (base,role) not in path_of:
                    p=os.path.join(tmpd,f"{ti}_{idx}_{role}.mp3"); open(p,"wb").write(g[role]); path_of[(base,role)]=p; idx+=1
        keys=list(path_of.keys()); code_of={}
        for i in range(0,len(keys),a.batch):
            chunk=keys[i:i+a.batch]
            codes=encode_paths([path_of[k] for k in chunk])
            for k,c in zip(chunk,codes): code_of[k]=c
        for base,g in groups.items():
            for spec in build_specs(base,g,bucket,full_keys,focus_keys,text_keys):
                tc=code_of.get((base,spec["audio"]))
                if tc is None or tc.shape[0]<4: continue
                rec={"id":f"{bucket}__{spec['idbase']}","text":spec["text"],"instruction":spec["instr"],
                     "language":spec["lang"],"tokens":int(tc.shape[0]),"quality":"high quality",
                     "audio_codes":tc.to(torch.int32).tolist(),"bucket":spec["bucket"]}
                if spec["ref_from"]:
                    rc=code_of.get((base,spec["ref_from"]))
                    if rc is None or rc.shape[0]<4: continue
                    rec["reference_audio_codes"]=[rc.to(torch.int32).tolist()]
                fout.write(json.dumps(rec,ensure_ascii=False)+"\n")
                counts[spec["bucket"]]=counts.get(spec["bucket"],0)+1
        fout.flush()
        for p in path_of.values():
            try: os.remove(p)
            except Exception: pass
        print(f"[{bucket}] tar {ti+1}/{len(tars)} ({tarfn}) total={sum(counts.values())}",flush=True)
    fout.close()
    print("COUNTS "+json.dumps(counts),flush=True)

if __name__=="__main__": main()
