#!/usr/bin/env python3
"""Combine per-source re-tokenized JSONL buckets into N training shards (one per data-parallel rank).
Dedups, length-filters (8-600 frames), and CHUNKS over-long clips from selected buckets into
<=600-frame pieces (instead of dropping them). Point the globs at your own data/ directory."""
import json, glob, random, os
random.seed(2026)
N=int(os.environ.get("N_SHARDS","7")); MINF,MAXF=8,600
DATA=os.environ.get("DATA_DIR","./data"); SHARDS=os.environ.get("SHARD_DIR","./shards")
os.makedirs(SHARDS,exist_ok=True)
# Buckets whose over-long clips should be chunked rather than dropped (name them to match your data).
CHUNK_BUCKETS=set((os.environ.get("CHUNK_BUCKETS","") or "").split(",")) - {""}

srcs=sorted(glob.glob(f"{DATA}/mix2_*.jsonl") + glob.glob(f"{DATA}/mix1m_*.jsonl") + glob.glob(f"{DATA}/pilot_*.jsonl"))

recs=[]; counts={}; seen=set(); chunked=0
def add(r):
    b=r.get("bucket","?"); counts[b]=counts.get(b,0)+1; recs.append(r)

for f in srcs:
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        ac=r.get("audio_codes")
        if not ac or len(ac[0])!=32: continue
        if not r.get("text") or not r.get("instruction"): continue
        fr=len(ac)
        if fr<MINF: continue
        key=(r.get("id"),fr,tuple(ac[0]))
        if key in seen: continue
        seen.add(key)
        if fr<=MAXF:
            add(r)
        elif r.get("bucket") in CHUNK_BUCKETS:
            nchunks=(fr+MAXF-1)//MAXF
            for ci in range(nchunks):
                seg=ac[ci*MAXF:(ci+1)*MAXF]
                if len(seg)<MINF: continue
                rr=dict(r); rr["audio_codes"]=seg; rr["tokens"]=len(seg); rr["id"]=f"{r.get('id')}_chunk{ci}"
                add(rr); chunked+=1
        # else: drop over-long non-chunk buckets

random.shuffle(recs)
outs=[open(f"{SHARDS}/mixexp.rank{i}-of-{N}.jsonl","w") for i in range(N)]
for i,r in enumerate(recs): outs[i%N].write(json.dumps(r,ensure_ascii=False)+"\n")
for o in outs: o.close()
json.dump(counts,open(f"{DATA}/bucket_counts_exp.json","w"),indent=2)
mf=sum(len(r["audio_codes"]) for r in recs)//max(len(recs),1)
json.dump({"total":len(recs),"mean_frames":mf,"chunked_pieces":chunked,"buckets":counts},
          open(f"{DATA}/mixexp_meta.json","w"),indent=2)
print("EXPANDED total unique records:",len(recs),"| chunked pieces:",chunked,"| mean_frames:",mf)
print(json.dumps(dict(sorted(counts.items(),key=lambda x:-x[1])),indent=2))
