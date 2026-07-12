#!/usr/bin/env bash
# Resume from a mid-run checkpoint and finish the remaining epochs, CONTINUING the LR schedule:
# start at the peak LR value where the original linear decay left off (example: ~5.15e-6), NO warmup,
# linear decay to 0. Same config otherwise. Checkpoint every epoch. Output dir is NEW so the
# checkpoint-epoch-K here map to the overall epochs (offset by however many you already ran).
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
GLOBAL=112; EPOCHS=${EPOCHS:-2}
SPE=$($PY -c "print(max(1,$N//$GLOBAL))")
TOTAL=$(( EPOCHS * SPE ))
RESUME_LR=${RESUME_LR:-5.15e-6}
echo "[resume] N=$N remaining_epochs=$EPOCHS steps/epoch=$SPE total_steps~$TOTAL warmup=0 lr=$RESUME_LR from=$RESUME_CKPT"
cd "${FINETUNE_DIR:?set FINETUNE_DIR to the moss_tts finetuning dir}"
$ENVBIN/accelerate launch --config_file "$(dirname "$0")/accelerate_zero3_1m.yaml" \
    sft.py \
    --model-path "${RESUME_CKPT:?set RESUME_CKPT to the checkpoint to continue from}" \
    --codec-weight-dtype fp32 --codec-compute-dtype bf16 --attn-implementation eager \
    --train-jsonl "$FILES" \
    --output-dir ./out/train_final \
    --per-device-batch-size 4 --gradient-accumulation-steps 4 \
    --learning-rate $RESUME_LR --warmup-steps 0 --lr-scheduler-type linear \
    --weight-decay 0.1 --max-grad-norm 1.0 \
    --num-epochs $EPOCHS --save-every-epochs 1 --logging-steps 5 \
    --mixed-precision bf16 --channelwise-loss-weight 1,64 \
    --gradient-checkpointing --gradient-checkpointing-impl model_internal --skip-nonfinite-batches
echo "TRAIN_FINAL_DONE rc=$?"
