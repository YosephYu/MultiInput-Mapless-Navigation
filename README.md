# Target-driven multi-input mapless robot navigation with deep reinforcement learning

The official implementation of [project](https://github.com/YosephYu/MultiInput-Mapless-Navigation) and [paper](https://iopscience.iop.org/article/10.1088/1742-6596/2513/1/012004).

![Pipeline](assets/overall.svg)

## Install

The latest codes are tested on Ubuntu 20.04, CUDA11.6, PyTorch 1.12.1 and Python 3.8.

The simulation environments are based on CoppeliaSim Version 4.4.0.

Dependencies are listed in `requirements.txt`:

```
pip install -r requirements.txt
```

## Train

Before training and testing, you should activate CoppeliaSim first.

```bash
bash CoppeliaSim/coppeliaSim.sh
```

Then load maps through `File > Open scene...` and select the map from `./scenes/`.

```bash
python train.py # edit config.py to match the map you select
```

## Test

```bash
python test.py # edit config.py to match the map you select
```

## Citation

If you find this repo useful in your research, please consider citing it:

```
@article{Xing_2023,
doi = {10.1088/1742-6596/2513/1/012004},
url = {https://dx.doi.org/10.1088/1742-6596/2513/1/012004},
year = {2023},
month = {jun},
publisher = {IOP Publishing},
volume = {2513},
number = {1},
pages = {012004},
author = {Hua Xing and Longfeng Yu},
title = {Target-driven multi-input mapless robot navigation with deep reinforcement learning},
journal = {Journal of Physics: Conference Series}
}
```

## Thanks

- [CoppeliaSim](https://www.coppeliarobotics.com/)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [marooncn/RL](https://github.com/marooncn/RL/tree/master)
- [AgrawalAmey/rl-car](https://github.com/AgrawalAmey/rl-car/tree/master)