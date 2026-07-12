# Checkpoint — MOSS-TTS-v1.5 8B voice-acting

The trained 8B checkpoint (~16 GB) is distributed as **GitHub Release assets**, split into
~1.9 GB zip parts (GitHub caps a single asset well below the full size). The parts are **not**
committed to this git repo — download them from the Release attached to this project.

Parts and their sha256 are listed in [`PARTS.md`](PARTS.md).

## Download

Download **every** part into one empty folder — the split set is
`moss8b_checkpoint.z01, moss8b_checkpoint.z02, …, moss8b_checkpoint.zip` (the final `.zip` is the
last part and holds the archive's central directory; you need all of them).

```bash
# from the Release page, or with the GitHub CLI:
gh release download <tag> -R <owner>/<repo> -D checkpoint_parts
cd checkpoint_parts
```

(Optional) verify integrity against `PARTS.md`:

```bash
sha256sum -c <(grep -E '  moss8b_checkpoint' PARTS.md)   # or check each part manually
```

## Reassemble & extract

```bash
# merge the split parts back into a single archive, then unzip
zip -s 0 moss8b_checkpoint.zip --out combined.zip
unzip combined.zip           # -> ./checkpoint-last/
```

This yields a `checkpoint-last/` directory with `model.safetensors`, the tokenizer/processor
files, and the model code (`modeling_moss_tts.py`, `processing_moss_tts.py`, etc.).

## Load

```python
import torch
from transformers import AutoModel, AutoProcessor

CKPT = "checkpoint-last"   # the extracted directory
proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
proc.audio_tokenizer = proc.audio_tokenizer.to("cuda").eval()
model = AutoModel.from_pretrained(CKPT, trust_remote_code=True,
                                  dtype=torch.bfloat16,
                                  attn_implementation="sdpa").to("cuda").eval()
```

Then generate with the helpers in [`../inference/`](../inference/). If you'd rather not download
the parts, the same weights are on the Hub:
[`laion/moss-tts-v1.5-8b-voice-acting`](https://huggingface.co/laion/moss-tts-v1.5-8b-voice-acting).
