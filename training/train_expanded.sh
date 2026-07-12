#!/usr/bin/env bash
# Continue from a prior epoch checkpoint for a few more epochs on an expanded mixture.
# Same config: ZeRO-3 + CPU-Adam, bf16, grad-checkpointing, peak LR 1e-5 + ~3% warmup + linear,
# global batch 112 (micro 4 x 7 x accum 4), audio channels weighted, skip-nonfinite, max-grad-norm 1.0.
# Checkpoint every epoch.
set +e
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your environment}
export USE_DS_CPU_ADAM=1 DS_SKIP_CUDA_CHECK=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
ENVBIN=${ENVBIN:-$(dirname "$(command -v python)")}
PY=$ENVBIN/python
SHARD_DIR=${SHARD_DIR:-./shards}
FILES=$(ls "$SHARD_DIR"/mixexp.rank*-of-7.jsonl | paste -sd, -)
N=$($PY -c "import glob;print(sum(1 for f in glob.glob('$SHARD_DIR/mixexp.rank*-of-7.jsonl') for _ in open(f)))")
GLOBAL=112; EPOCHS=${EPOCHS:-4}
SPE=$($PY -c "print(max(1,$N//$GLOBAL))")
TOTAL=$(( EPOCHS * SPE ))
WARMUP=$($PY -c "print(max(5,int(0.03*$TOTAL)))")
echo "[exp] N=$N epochs=$EPOCHS steps/epoch=$SPE total_steps~$TOTAL warmup=$WARMUP resume=$RESUME_CKPT"
cd "${FINETUNE_DIR:?set FINETUNE_DIR to the moss_tts finetuning dir}"
$ENVBIN/accelerate launch --config_file "$(dirname "$0")/accelerate_zero3_1m.yaml" \
    sft.py \
    --model-path "${RESUME_CKPT:?set RESUME_CKPT to the checkpoint to continue from}" \
    --codec-weight-dtype fp32 --codec-compute-dtype bf16 --attn-implementation eager \
    --train-jsonl "$FILES" \
    --output-dir ./out/train_expanded \
    --per-device-batch-size 4 --gradient-accumulation-steps 4 \
    --learning-rate 1.0e-5 --warmup-steps $WARMUP --lr-scheduler-type linear \
    --weight-decay 0.1 --max-grad-norm 1.0 \
    --num-epochs $EPOCHS --save-every-epochs 1 --logging-steps 5 \
    --mixed-precision bf16 --channelwise-loss-weight 1,64 \
    --gradient-checkpointing --gradient-checkpointing-impl model_internal --skip-nonfinite-batches
echo "TRAIN_EXPANDED_DONE rc=$?"
