# Fast inference — MOSS-TTS-v1.5 8B

How to get the most speech per GPU-second out of the 8B voice-acting model. Short version:
**single-stream decoding is fixed at ~2× realtime; batching is the entire lever.**

## Measured throughput

One A100/H100-class 80 GB GPU, bf16, `attn_implementation="sdpa"`, 200 generated tokens per clip
(~13 s of audio each). Same prompt replicated to batch size B (`throughput_benchmark.py`):

| Batch B | clips/s | tokens/s | RTF (audio-s / wall-s) | peak VRAM (GB) |
|--------:|--------:|---------:|-----------------------:|---------------:|
| 1  | 0.161 | 32.1   | 2.16×  | 24.3 |
| 2  | 0.301 | 60.3   | 3.99×  | 24.4 |
| 4  | 0.589 | 117.9  | 7.02×  | 24.6 |
| 8  | 1.149 | 229.8  | 13.04× | 25.0 |
| 16 | 2.202 | 440.3  | 26.05× | 25.9 |
| 24 | 3.428 | 685.6  | 43.49× | 26.7 |
| 32 | 4.267 | 853.4  | 54.41× | 27.5 |
| 48 | 4.654 | 930.9  | 57.02× | 29.1 |
| 64 | 5.452 | 1090.4 | 66.93× | 30.7 |

## Analysis

- **Single stream is ~32 decode-steps/s ≈ 2× realtime, and that's architecturally fixed.** Each
  step emits one frame of the 32-codebook delay grid; the autoregressive dependency means B=1
  latency won't improve much regardless of tuning. At ~18 audio frames/s of output, 32 steps/s is
  roughly 2× faster than realtime — fine for latency, poor for bulk throughput.

- **Batching is the lever.** Going B=1 → B=64 gives **~34× throughput** (0.16 → 5.45 clips/s;
  RTF 2× → 67×; tokens/s 32 → 1090). The decode is memory-bandwidth/latency bound at small B, so
  extra sequences ride along almost for free until compute saturates.

- **VRAM is not the constraint.** Even at B=64 the model uses only **~30 GB** of an 80 GB card, so
  there's ample headroom to push **B ≈ 128–256** — extrapolating the curve, roughly **8–10 clips/s
  per GPU** before memory pressure. (Beyond ~B=48 the per-step cost starts rising, so the curve
  bends; measure on your hardware.)

- **flash-attention gives essentially NO speedup here — measured.** We benchmarked
  `attn_implementation="flash_attention_2"` (flash-attn 2.7.4) against `sdpa` on the same GPU:

  | batch | sdpa clips/s | flash-attn clips/s | speedup | peak VRAM |
  |------:|-------------:|-------------------:|--------:|----------:|
  |     1 |        0.146 |              0.142 |   0.97× | 24.3 GB (identical) |
  |     8 |        1.105 |              1.103 |   1.00× | 25.0 GB |
  |    32 |        4.096 |              3.753 |   0.92× | 27.5 GB |
  |    64 |        5.202 |              5.237 |   1.01× | 30.7 GB |

  Flash-attention only helps when attention (O(seq²)) dominates — i.e. long contexts. MOSS generation
  is short-sequence autoregressive decoding (prompt + ~200 frames), so attention is a tiny slice of
  the per-step cost; the bottleneck is the 8B transformer forward + KV-cache access. Result: **no
  throughput gain and no memory saving.** `sdpa` (the default) is all you need — **batching is the
  only real lever.** The scripts still auto-detect flash-attn but it is not required.

- **Multi-GPU ≈ linear.** Generation is embarrassingly parallel across data — shard prompts over
  GPUs (one model replica each). ~6 GPUs ≈ **50–60 clips/s** aggregate.

- **vLLM / SGLang?** They would add continuous batching, paged-KV, and CUDA graphs on top of this.
  But MOSS uses a **custom 32-codebook `MossTTSDelay` head** (32 audio logits + 1 text logit per
  step, with the delay pattern), which is not a standard causal-LM head, so it needs a **custom
  backend adapter** and is **not supported upstream** today. Until such an adapter exists, plain
  `transformers` + large static batches is the practical fast path.

## Practical recommendation

- **Batch 48–64 per GPU** for bulk generation — near-peak throughput at modest VRAM.
- Use best-of-N by replicating one prompt to fill the batch (`batched.py`), then keep the
  top-scoring take with your quality/genuineness scorer.
- Don't bother with `flash_attn` — it gives no measurable speedup for this short-sequence decode
  workload (see table above). `sdpa` is the default and is all you need. Shard across GPUs for
  linear scale-out.
- Keep `audio_top_p=0.95`, `audio_top_k=25`, `audio_repetition_penalty=1.1`; set
  `audio_temperature` to 1.0 with a reference clip or 0.8 without (see the root README grid-search
  settings).
