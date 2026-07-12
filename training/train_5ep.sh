#!/usr/bin/env bash
# Full fine-tune for N epochs over your training mixture.
# Config: DeepSpeed ZeRO-3 + CPU-Adam, peak lr 1e-5 + ~3% warmup + linear decay,
# global batch 112 (micro 4 x 7 GPUs x accum 4), bf16, grad-checkpointing,
# skip-nonfinite batches, per-channel CE (audio channels weighted). Checkpoint every epoch.
#
# DATA: point --train-jsonl at your own retokenized shards. Each JSONL line is one sample:
#   {"text", "instruction", "language", "tokens", "quality", "audio_codes":[[.. 32 ..], ...],
#    optional "reference_audio_codes":[[[..32..], ...]]}
# See retokenize*.py for how to turn a WebDataset / directory of {audio_path, text, instruction}
# samples into these shards.
set +e
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in your environment}
export USE_DS_CPU_ADAM=1 DS_SKIP_CUDA_CHECK=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
ENVBIN=${ENVBIN:-$(dirname "$(command -v python)")}
PY=$ENVBIN/python
# Your retokenized training shards, one per data-parallel rank (7 shards for 7 GPUs).
SHARD_DIR=${SHARD_DIR:-./shards}
FILES=$(ls "$SHARD_DIR"/mix.rank*-of-7.jsonl | paste -sd, -)
N=$($PY -c "import glob;print(sum(1 for f in glob.glob('$SHARD_DIR/mix.rank*-of-7.jsonl') for _ in open(f)))")
GLOBAL=112
EPOCHS=${EPOCHS:-5}
STEPS_PER_EPOCH=$($PY -c "print(max(1,$N//$GLOBAL))")
TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
WARMUP=$($PY -c "print(max(5,int(0.03*$TOTAL_STEPS)))")
echo "[train] N=$N epochs=$EPOCHS steps/epoch=$STEPS_PER_EPOCH total_steps~$TOTAL_STEPS warmup=$WARMUP"
cd "${FINETUNE_DIR:?set FINETUNE_DIR to the moss_tts finetuning dir}"
$ENVBIN/accelerate launch --config_file "$(dirname "$0")/accelerate_zero3_1m.yaml" \
    sft.py \
    --model-path "${MODEL_PATH:?set MODEL_PATH to the base MOSS-TTS-v1.5 8B checkpoint}" \
    --codec-weight-dtype fp32 --codec-compute-dtype bf16 --attn-implementation eager \
    --train-jsonl "$FILES" \
    --output-dir ./out/train5ep \
    --per-device-batch-size 4 --gradient-accumulation-steps 4 \
    --learning-rate 1.0e-5 --warmup-steps $WARMUP --lr-scheduler-type linear \
    --weight-decay 0.1 --max-grad-norm 1.0 \
    --num-epochs $EPOCHS --save-every-epochs 1 --logging-steps 5 \
    --mixed-precision bf16 --channelwise-loss-weight 1,64 \
    --gradient-checkpointing --gradient-checkpointing-impl model_internal --skip-nonfinite-batches
echo "TRAIN_5EP_DONE rc=$?"
