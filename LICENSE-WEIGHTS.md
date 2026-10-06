# Licence of the model weights

| Artefact | Licence |
|---|---|
| Code in this repository | Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)) |
| GeoCR weights on the Hub: `transformer/`, `stems/`, `lora/` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `vae/ae.safetensors` | Apache-2.0, the FLUX.2 autoencoder of [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) |
| Comparison-method checkpoints under `baselines/` on the Hub | the terms of each method's upstream code; see [`baselines/README.md`](https://huggingface.co/JeonghyeokDo/GeoCR/blob/main/baselines/README.md) |
| Datasets | not redistributed; each is subject to its distributor's terms ([DATA.md](DATA.md)) |

The GeoCR transformer was initialised from FLUX.2 [klein] 4B Base (Apache-2.0) and pretrained on the training
splits of ten datasets, including AllClear, which is distributed under CC BY-NC 4.0. The GeoCR weights are
therefore released for non-commercial use.

The DINO similarity metric uses the DINOv3-SAT ViT-L/16 weights, which are distributed by Meta under the DINOv3
License and are not included in this release.
