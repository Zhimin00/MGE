# Evaluation

## Get started

Create the reproducible Conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate pi3
```

If the environment already exists, update it after pulling new changes:

```bash
conda env update -f environment.yml --prune
```

## Overview

- [x] Video Depth Estimation
- [x] Relative Camera Pose Estimation
- [x] Multi-view Reconstruction (Point Map Estimation)
- [x] Doppelganger Camera-Pose Benchmark

The root config file of all evaluations is `configs/eval.yaml`, however you don't need to edit it

- All main hyperparameters you need are in `configs/evaluation/xxxxx.yaml`
- Sometimes you may want to change the dataset config in `configs/data/xxxxx.yaml`, or the model config in `configs/model/xxxxx.yaml`

## Dataset Preparation

Please put all evaluation datasets under `data` folder, or you can change the config in `configs/data/xxxxx.yaml`.


## 1. Video Depth Estimation

configs in `configs/evaluation/videodepth.yaml`, see [videodepth/README.md](videodepth/README.md) for more details.

```bash
python videodepth/infer.py
python videodepth/eval.py
```

## 2. Relative Camera Pose Estimation


```bash
python relpose/eval_dist.py
```

## 3. Multi-view Reconstruction (Point Map Estimation)


```bash
# python mv_recon/sampling.py  # to generate seq-id-maps under datasets/seq-id-maps, which is provided in this repo
python mv_recon/eval.py
```

For token merging inference

```bash
python mv_recon/eval_aga.py
```

## 4. Doppelganger Camera-Pose Benchmark

The `doppelganger_benchmark/` directory contains 43 fixed mixed-input subsets from eight
outdoor scenes. Evaluation scripts and camera metadata are tracked on this GitHub branch;
the 860 input images are hosted on [Hugging Face](https://huggingface.co/shaozhimin/MGE).

Download the images from the root of this repository:

```bash
pip install -U "huggingface_hub[cli]"
hf download shaozhimin/MGE \
  --include "doppelganger_benchmark/subsets/*/*/images/*.jpg" \
  --local-dir .
```

See the [Doppelganger benchmark README](doppelganger_benchmark/README.md) for the dataset
layout, Pi3 inference command, prediction format, and camera-pose evaluation protocol.

## Demo
```bash
python demo_gradio.py
```

## Acknowledgement

Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r)
- [Spann3R](https://github.com/HengyiWang/spann3r)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [MoGe](https://github.com/microsoft/MoGe)
- [VGGT](https://github.com/facebookresearch/vggt)
- [FastVGGT](https://github.com/mystorm16/FastVGGT)

<!-- ## Citation -->

<!-- If you find our work useful, please consider citing:

```bibtex
@misc{wang2025pi3,
      title={$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning}, 
      author={Yifan Wang and Jianjun Zhou and Haoyi Zhu and Wenzheng Chang and Yang Zhou and Zizun Li and Junyi Chen and Jiangmiao Pang and Chunhua Shen and Tong He},
      year={2025},
      eprint={2507.13347},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13347}, 
}
``` -->


<!-- ## License
For academic use, this project is licensed under the 2-clause BSD License. See the [LICENSE](./LICENSE) file for details. For commercial use, please contact the authors. -->
