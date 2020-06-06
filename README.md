# DSAC
Implementation of Distributional Soft Actor Critic (DSAC).
This repository is based on [RLkit](https://github.com/vitchyr/rlkit), a reinforcement learning framework implemented by PyTorch.
The core algorithm of DSAC is in `rlkit/torch/dsac/`

## Requirements
- python 3.6+
- pytorch 1.0+
- gym[all] 0.15+ 
- scipy 1.0+
- numpy
- matplotlib
- gtimer
- pyyaml

## Usage
You can write your experiment settings in YAML and run with 
```
python dsac.py --config your_config.yaml --gpu 0 --seed 0
```
To run our implementation of SAC/TD3/TD4, please replace dsac.py with sac.py/td3.py/td4.py. Set `--gpu -1`, your program will run on CPU.

The experimental configurations of the paper are in `config/`. A typical configuration in YAML is given as follow:
```
env: Hopper-v2
version: normal-iqn-neutral # version for logging
eval_env_num: 10 # # of paralleled environments for evaluation
expl_env_num: 10 # of paralleled environments for exploration
layer_size: 256 # hidden size of networks
num_quantiles: 32
replay_buffer_size: 1000000
algorithm_kwargs:
  batch_size: 256
  max_path_length: 1000
  min_num_steps_before_training: 10000
  num_epochs: 1000
  num_eval_paths_per_epoch: 10
  num_expl_steps_per_train_loop: 1000
  num_trains_per_train_loop: 1000
trainer_kwargs:
  alpha: 0.2
  discount: 0.99
  policy_lr: 0.0003
  zf_lr: 0.0003
  soft_target_tau: 0.005
  tau_type: iqn # quantile fraction generation method, choices: fix, iqn, fqf
  use_automatic_entropy_tuning: false
```

Learning under risk measures is available for DSAC and TD4. We provide 6 choices of risk metrics: `neutral`, `std`, `VaR`, `cpw`, `wang`, `cvar`. You can change the risk preference by add two additional items in your YAML config:
```
...

trainer_kwargs:
  ...
  risk_type: std
  risk_param: 0.1
```


