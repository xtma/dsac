# DSAC
Implementation of Distributional Soft Actor Critic (DSAC).
This repositorie is based on [RLkit](https://github.com/vitchyr/rlkit), a reinforcement learning framework implemented in PyTorch.
Core algorithm of DSAC is in `rlkit/torch/dsac/`

## Requirements
- python 3.6+
- pytorch 1.0+
- gym[all] 0.15+ 
- [highway](https://github.com/eleurent/highway-env) 1.0
- scipy 1.0+
- numpy
- matplotlib
- gtimer

## Usage
You can write your experiment settings in YAML and run with 
```
python dsac.py --config your_config.yaml --gpu 0 --seed 0
```
Set `--gpu -1`, your program will run on CPU.

All the configs we use are in `config/`. To run our implementation of SAC or TD3, please replace dsac.py with sac.py or td3.py.