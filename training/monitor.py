#!/usr/bin/env python3
"""Tail a MOSS sft.py training log and write dash/status.json for a live dashboard."""
import re, json, time, os, sys, glob, subprocess, argparse

LINE=re.compile(r"epoch=(\d+) step=(\d+)/(\d+) loss=([0-9.]+) lr=([0-9.eE+-]+).*?samples_per_sec=([0-9.]+)")

def fmt_dur(s):
    s=int(max(s,0)); h=s//3600; m=(s%3600)//60; sec=s%60
    if h: return f"{h}h {m}m"
    if m: return f"{m}m {sec}s"
    return f"{sec}s"

def gpu_mem():
    try:
        out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
        vals=[int(x) for x in out.split()]
        return max(vals[:7])/1024.0 if vals else 0
    except Exception: return 0

def newest_ckpt(d):
    cks=glob.glob(os.path.join(d,"checkpoint-*"))
    if not cks: return "none yet"
    cks.sort(key=lambda p:os.path.getmtime(p))
    return os.path.basename(cks[-1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log",required=True); ap.add_argument("--status",default="./dash/status.json")
    ap.add_argument("--phase",required=True); ap.add_argument("--global-batch",type=int,required=True)
    ap.add_argument("--target-samples",type=int,required=True); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--buckets-json",default="./data/bucket_counts.json")
    ap.add_argument("--avg-tokens",type=int,default=420); ap.add_argument("--note",default="")
    a=ap.parse_args()
    curve=[]; last_step=-1
    while True:
        step=maxs=0; loss=None; sps=0.0
        try:
            txt=open(a.log,errors="ignore").read()[-200000:]
            ms=LINE.findall(txt)
            if ms:
                e,st,mx,ls,lr,sp=ms[-1]
                step=int(st); maxs=int(mx); loss=float(ls); sps=float(sp)
        except Exception: pass
        if loss is not None and step!=last_step:
            curve.append([step,round(loss,4)]); last_step=step
            if len(curve)>400: curve=curve[::2]
        seen=step*a.global_batch
        eta=fmt_dur((a.target_samples-seen)/max(sps,1e-6)) if sps>0 else "--"
        bk={}
        try: bk=json.load(open(a.buckets_json))
        except Exception: pass
        s={"phase":a.phase,"step":step,"total_steps":maxs,"samples_seen":seen,
           "target_samples":a.target_samples,"loss":loss,"loss_curve":curve,
           "tokens_per_s":round(sps*a.avg_tokens),"throughput_samples_s":round(sps,2),
           "gpu_mem_gb":round(gpu_mem(),1),"eta":eta,"last_checkpoint":newest_ckpt(a.out_dir),
           "bucket_counts":bk,"updated":time.strftime("%Y-%m-%d %H:%M:%S"),"note":a.note}
        # multi-emotion sample manifest (regenerated each epoch)
        smf=os.path.join(os.path.dirname(a.status),"samples.json")
        if os.path.exists(smf):
            try:
                m=json.load(open(smf)); s["samples"]=m.get("samples",[]); s["samples_tag"]=m.get("tag","")
            except Exception: pass
        tmp=a.status+".tmp"; json.dump(s,open(tmp,"w")); os.replace(tmp,a.status)
        time.sleep(5)

if __name__=="__main__": main()
