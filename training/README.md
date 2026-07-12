# Training — MOSS-TTS-v1.5 8B voice-acting fine-tune

This directory holds the full recipe used to fine-tune **MOSS-TTS-v1.5 8B**
(a Qwen3-8B backbone with a 32-codebook delay audio head, `MossTTSDelay`, operating on
24 kHz audio codes) for expressive, instruction-controlled voice acting. It contains the
retokenization pipeline, the DeepSpeed/Accelerate configs, the launchers, and small
smoke/monitor utilities. **No dataset is included or referenced** — plug in your own corpus.

## The model in one paragraph

MOSS-TTS-v1.5 encodes speech with a neural codec into a `(T, 32)` grid of discrete codes
(T frames, 32 codebooks per frame). The language model predicts, per position, one text token
(channel 0) plus 32 audio codes (channels 1..32) under a **delay** pattern so codebook *k* is
offset by *k* steps — the `MossTTSDelay` head. A style **instruction** ("a heartbroken woman,
voice trembling…"), the **text** to speak, and an optional **reference** clip (for voice cloning)
are formatted into a chat-style prompt; the model then generates the audio-code grid, which the
codec decodes back to a waveform.

## Data format (bring your own)

The trainer consumes JSONL shards, one sample per line:

```json
{"text": "...", "instruction": "...", "language": "English", "tokens": 173,
 "quality": "high quality", "audio_codes": [[c0, c1, ..., c31], ...],
 "reference_audio_codes": [[[...32...], ...]]   // optional, for voice-clone pairs
}
```

You produce these by **retokenizing** a WebDataset (tar shards) or a directory of
`{audio_path, text, instruction}` samples: encode each audio clip to `(T, 32)` codes with the
model's own codec (`AutoProcessor.encode_audios_from_path`), then write the record above.

- `retokenize.py` — local tar shards → JSONL (one source per GPU).
- `retokenize_hf.py` — stream remote WebDataset tars → JSONL.
- `retokenize_expand.py` — paired (target + reference) sources → several records each
  (standalone + voice-clone).
- `launch_expand.sh` — fan the retokenizers out across GPUs.
- `combine_expanded.py` — dedup, length-filter (8–600 frames), chunk over-long clips, and shard
  into one JSONL per data-parallel rank.

Set `MODEL_PATH` to the base MOSS-TTS-v1.5 8B checkpoint and `HF_TOKEN` in your environment
(never commit a token). Replace the `<YOUR_DATASET_*>` placeholders with your own sources.

## Loss construction

Each example is built as `[user prompt] + [assistant audio]` in `computing_loss` mode
(`processor(..., mode="computing_loss")`), then labels are masked so **only the audio target is
supervised**:

- mask the **entire user prompt** (`labels[:, :P, :] = -100`),
- mask **padding** positions (`labels[~attention_mask] = -100`),
- mask the **audio-pad code** (1024) in each audio channel and the **text-pad** id in channel 0
  (their logits are forced to `-inf`, so they must not be targets).

Loss is per-channel cross-entropy combined with `channelwise_loss_weight` (e.g. `1,64` — a small
weight on the text channel, a larger aggregate weight across the 32 audio channels). `smoke.py`
proves the masking is correct: it overfits a single sample and the loss must collapse.

## Optimizer / distributed setup

- **DeepSpeed ZeRO-3** with **CPU-Adam** optimizer offload (`ds_zero3_1m.json`,
  `accelerate_zero3_1m.yaml`) — fits the 8B model + optimizer state across 7 GPUs + host RAM.
- **bf16** mixed precision; **gradient checkpointing** (`model_internal` impl).
- Global batch **112** (micro-batch 4 × 7 GPUs × grad-accum 4).
- **LR 1e-5** peak, ~3% linear warmup, linear decay to 0; weight decay 0.1; max-grad-norm 1.0.
- Skip non-finite batches; checkpoint every epoch.
- Codec kept in fp32 weights / bf16 compute; attention `eager` during training.

## Launchers

- `train_5ep.sh` — full fine-tune for **N epochs** (default 5) from the base checkpoint.
- `train_expanded.sh` — continue from a prior epoch checkpoint on an expanded mixture (fresh warmup).
- `train_resume.sh` — resume mid-schedule: **continue the LR decay** from where it left off
  (`RESUME_LR`, no warmup) to finish the remaining epochs.

Set `MODEL_PATH` / `RESUME_CKPT`, `FINETUNE_DIR` (the `moss_tts` finetuning dir with `sft.py`),
`SHARD_DIR`, and `HF_TOKEN`, then run the launcher. `monitor.py` tails the training log into a
`status.json`; `gen_sample.py` renders a few held-out emotional prompts from any checkpoint.

> The exact epoch count, mixture, and corpus size are yours to choose — none are prescribed here.
