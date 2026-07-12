# Checkpoint parts — MOSS-TTS-v1.5 8B voice-acting

Split-zip of `checkpoint-last/` (store-only, `zip -s 1900m -0`). **9 parts, 16,995,770,068 bytes total (~17.00 GB).**

Download **all** parts into one folder (including the final `.zip`, which holds the archive central directory), then reassemble — see `README.md`.

| part | size (bytes) | sha256 |
|---|---:|---|
| `moss8b_checkpoint.z01` | 1,992,294,400 | `ece494e2ec3712effaf703294942c14be52b224165389643a8835710a3759370` |
| `moss8b_checkpoint.z02` | 1,992,294,400 | `cb95d8d1d12a26c408b0d106da3ac3827f9b34abc9aa43209b399ef96f54b46e` |
| `moss8b_checkpoint.z03` | 1,992,294,400 | `a3b2282d7c7917822be2e775ba9d60050a1c0d954c60ab955740dd5e750775c5` |
| `moss8b_checkpoint.z04` | 1,992,294,400 | `c036e6e97179fc28dde0050b15bd2212193190a0b0e487f17bed1bdc58922574` |
| `moss8b_checkpoint.z05` | 1,992,294,400 | `24f93094bd7acfec16d91220135f562ada2d678086445091e4b3d065df4128df` |
| `moss8b_checkpoint.z06` | 1,992,294,400 | `b6053ae410ca9385d5bbfdc4075f1e5d69269c55f1b1bb8cc51f8c7fa24e6b3f` |
| `moss8b_checkpoint.z07` | 1,992,294,400 | `b5a51dafd210ae97967d3206fd3dd04c46d07552b0ea438a29a17b2df64d7777` |
| `moss8b_checkpoint.z08` | 1,992,294,400 | `38b6ca07428526a57c3b6a39c39ecc00b6122a4d68949c99bd7cb03830ccd1b7` |
| `moss8b_checkpoint.zip` | 1,057,414,868 | `3b53a66debafb026bd22b715f91d97ff094f2acf93fffca9a9a14f45b1cc8584` |

## Verify

```bash
sha256sum -c PARTS.sha256   # if you extract the checksums, or check each row above
```

## Reassemble

```bash
zip -s 0 moss8b_checkpoint.zip --out combined.zip
unzip combined.zip   # -> checkpoint-last/
```
