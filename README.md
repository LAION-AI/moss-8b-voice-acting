# MOSS-TTS-v1.5 8B — Voice-Acting

Expressive, **instruction-controlled** text-to-speech. This repo is a **voice-acting fine-tune of
MOSS-TTS-v1.5 8B**: a Qwen3-8B backbone with a **32-codebook delay audio head** (`MossTTSDelay`)
operating on **24 kHz** audio codes. Give it a natural-language *performance instruction* ("a
heartbroken woman, voice trembling with tears…"), the *text* to speak, and — optionally — a
*reference clip* to clone a voice, and it renders emotive, human-sounding speech.

- **Model:** [`laion/moss-tts-v1.5-8b-voice-acting`](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting)
- **License:** Apache-2.0
- **What's here:** minimal + fast inference (`inference/`), the full fine-tune recipe
  (`training/`), a GitHub Pages demo/eval site (`docs/`), and a split-zip of the trained
  checkpoint for a Release (`checkpoint/`).

## Install

```bash
pip install "transformers>=4.44" torch torchaudio huggingface_hub
# optional, faster attention: pip install flash-attn --no-build-isolation
```

## Quickstart

**Single utterance** (`inference/single.py`):

```bash
python inference/single.py \
  --text "We did it. We actually did it!" \
  --instruction "An ecstatic young man, laughing between words, breathless with joy." \
  --out joy.wav
# voice cloning: add  --reference /path/to/speaker.wav
```

**Fast batched / best-of-N** (`inference/batched.py`) — one `generate` call returns many takes:

```bash
python inference/batched.py --text "Hello, world." --n 16 --outdir takes/
```

Both scripts load from the HF repo by default (`MOSS_MODEL` env var) or from a local checkpoint
directory, use bf16, and auto-select `flash_attention_2` if `flash_attn` is installed (else `sdpa`).

## Recommended generation settings

From a systematic grid search over temperature × repetition-penalty (scored on WER, a learned
blend-quality head, and genuineness). Keep `audio_top_p=0.95`, `audio_top_k=25`.

| Scenario | `audio_temperature` | `audio_repetition_penalty` |
|---|---|---|
| **With** a reference clip (voice cloning) | **1.0** | **1.1** |
| **Without** a reference (instruction only) | **0.8** | **1.1** |

Rule of thumb: higher temperature → more expressive/genuine; lower → more intelligible. Raise the
repetition penalty only if you hear buzzing/stuttering. See the
[grid-search grid](https://projects.laion.ai/moss-8b-voice-acting/gridsearch.html).

## Fast inference

Single-stream decoding is architecturally fixed at ~2× realtime; **batching is the entire lever**
(≈34× throughput from B=1 to B=64, at only ~30 GB VRAM). Measured on one 80 GB GPU, bf16, `sdpa`:

| Batch | clips/s | tokens/s | RTF | peak VRAM |
|------:|--------:|---------:|----:|----------:|
| 1  | 0.16 | 32   | 2.2×  | 24.3 GB |
| 8  | 1.15 | 230  | 13×   | 25.0 GB |
| 32 | 4.27 | 853  | 54×   | 27.5 GB |
| 64 | 5.45 | 1090 | 67×   | 30.7 GB |

Practical recommendation: **batch 48–64 per GPU**; shard across GPUs for near-linear scale-out.
Full table + analysis (flash-attn, multi-GPU, vLLM/SGLang caveats) in
[`inference/FAST_INFERENCE.md`](inference/FAST_INFERENCE.md).

## Training

The complete fine-tune recipe — retokenization pipeline, DeepSpeed ZeRO-3 + CPU-Adam configs,
launchers, and loss construction — is in [`training/`](training/README.md). Bring your own data
(a WebDataset / directory of `{audio_path, text, instruction}` samples); no dataset is shipped.

## Live demos & evals

Hosted on GitHub Pages (play the audio in-browser):

- **Samples:** [Voice-cloning + paraphrase](https://projects.laion.ai/moss-8b-voice-acting/paraphrase_clones.html)
- **Rankings:** [Genuineness (best-of-10)](https://projects.laion.ai/moss-8b-voice-acting/genuineness_ranking.html) ·
  [Combined reward](https://projects.laion.ai/moss-8b-voice-acting/reward_ranking.html)
- **Benchmarks:** [Generation-settings grid](https://projects.laion.ai/moss-8b-voice-acting/gridsearch.html) ·
  [Best-of-10 uplift](https://projects.laion.ai/moss-8b-voice-acting/eval_bestof10.html) ·
  [Best-of-32 uplift](https://projects.laion.ai/moss-8b-voice-acting/eval_bestof32.html)
- **Checkpoint eval:** [Held-out emotional prompts](https://projects.laion.ai/moss-8b-voice-acting/checkpoint_eval.html)
- **Prompting ablation (DramaBox):** [dialogue-in-instruction vs directions-only + emotion-reference](https://projects.laion.ai/moss-8b-voice-acting/dramabox_prompting_ablation.html) ·
  [top-3 grids](https://projects.laion.ai/moss-8b-voice-acting/dramabox_prompting_ablation_top3.html) ·
  [best-of-16/8/6/4 grids](https://projects.laion.ai/moss-8b-voice-acting/dramabox_prompting_ablation_bestofn.html) ·
  [prompt-adherence reward grids](https://projects.laion.ai/moss-8b-voice-acting/dramabox_prompting_ablation_promptreward.html)
- **Scaling & Sidon study (LGT):** [best-of-N audio samples](https://projects.laion.ai/moss-8b-voice-acting/lgt_bestof_samples.html) ·
  [report](https://projects.laion.ai/moss-8b-voice-acting/lgt_study_report.html)

Landing page: <https://projects.laion.ai/moss-8b-voice-acting/>

## Checkpoint download

The trained 8B checkpoint (~16 GB) is attached as **GitHub Release** assets, split into ~1.9 GB
zip parts. See [`checkpoint/README.md`](checkpoint/README.md) for the reassemble-and-load steps.

## Related models

- **This model:** [`laion/moss-tts-v1.5-8b-voice-acting`](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting)
- **Genuineness predictors:**
  [`laion/voiceclap-large-v2-genuineness`](https://huggingface.co/laion/voiceclap-large-v2-genuineness) ·
  [`laion/voiceclap-commercial-genuineness`](https://huggingface.co/laion/voiceclap-commercial-genuineness)
- **Blend quality scorer:** [`laion/voiceclap-vocalburst-blend`](https://huggingface.co/laion/voiceclap-vocalburst-blend)
- **Base VoiceCLAP embeddings:**
  [`laion/voiceclap-commercial`](https://huggingface.co/laion/voiceclap-commercial) ·
  [`laion/voiceclap-large-v2`](https://huggingface.co/laion/voiceclap-large-v2)

## Citation

See [`CITATION.cff`](CITATION.cff). Model and code are released under Apache-2.0.

- **Scaling study:** [throughput + best-of-N plateau + Sidon](https://projects.laion.ai/moss-8b-voice-acting/scaling_pilot/index.html) · [🎧 Top-3 samples per group](https://projects.laion.ai/moss-8b-voice-acting/scaling_pilot/samples.html)
- **Prompting ablation:** [with vs. without in-instruction lines + best-of-k scaling](https://projects.laion.ai/moss-8b-voice-acting/prompting_ablation/index.html) · [🎧 top-3 of 16](https://projects.laion.ai/moss-8b-voice-acting/prompting_ablation/grid_top3.html) · [best-of-8](https://projects.laion.ai/moss-8b-voice-acting/prompting_ablation/grid_bestof8.html)
