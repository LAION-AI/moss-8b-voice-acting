#!/usr/bin/env bash
# Fan out re-tokenization across N GPUs, one process per data source / tar slice.
# Each process streams a WebDataset (tar shards of {audio, text, instruction} samples),
# encodes the audio to 32-codebook MOSS codes, and writes a trainer-ready JSONL shard.
# Adapt the SOURCES / slices below to your own corpus, then combine with combine_expanded.py.
set +e
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your environment}
export TOKENIZERS_PARALLELISM=false
PY=${PY:-python}
cd "$(dirname "$0")"
mkdir -p data logs
rm -f data/mix2_*.jsonl logs/retok2_*.log

# Example: split one large source across 2 GPUs by tar index, and give each of the other
# sources its own GPU. Replace <YOUR_DATASET_*> and the tar counts with your own.
CUDA_VISIBLE_DEVICES=0 $PY retokenize_expand.py --source <YOUR_DATASET_A> --tar-start 0  --tar-end 47 --out data/mix2_a0.jsonl > logs/retok2_a0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY retokenize_expand.py --source <YOUR_DATASET_A> --tar-start 47 --tar-end 94 --out data/mix2_a1.jsonl > logs/retok2_a1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY retokenize_expand.py --source <YOUR_DATASET_B> --tar-start 0 --tar-end 4 --out data/mix2_b.jsonl > logs/retok2_b.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY retokenize_expand.py --source <YOUR_DATASET_C> --tar-start 0 --tar-end 4 --out data/mix2_c.jsonl > logs/retok2_c.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY retokenize_hf.py --source <YOUR_DATASET_D> --tars ALL --limit 0 --out data/mix2_d.jsonl > logs/retok2_d.log 2>&1 &
echo "all retokenize jobs launched; waiting..."
wait
echo "EXPAND_RETOK_DONE"
for f in logs/retok2_*.log; do echo "--- $(basename "$f") ---"; grep COUNTS "$f" | tail -1; done
wc -l data/mix2_*.jsonl
