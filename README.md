# PyTorch A2C, PPO and TRPO

This is a PyTorch implementation of:

- [Synchronous A3C (A2C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)

This is a fork of [this repository](https://github.com/Khrylx/PyTorch-RL).

## Important notes

- If you have a GPU, PyTorch will create additional threads when performing computations which can damage the performance of multiprocessing. This problem is most serious with Linux, where multiprocessing can be even slower than a single thread. You may have to set the OMP_NUM_THREADS to 1:
```
export OMP_NUM_THREADS=1
```

## Features

- Support CUDA (x10 faster than CPU implementation)
- Support multiprocessing for agent to collect samples in multiple environments simultaneously (x8 faster than single thread)
- Implement Fast Fisher vector product calculation

## Examples

A2C:

```
python examples/a2c_gym.py --env-name ...
```

PPO:

```
python examples/ppo_gym.py --env-name ...
```

TRPO:

```
python examples/trpo_gym.py --env-name ...
```