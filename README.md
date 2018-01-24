# Distributed-MADDPG
Master Course Graduation Project - Distributed Multi-Agent Cooperation Algorithm based on MADDPG with prioritized batch data.

## Distributed Multi-Agent Architecture
<p align="center">
  <img src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/architecture.PNG">
</p>

## Introduction
This work focus on Multi-Agent Cooperation Problem. We proposed a method which consists 3 components:
1. Related research - MADDPG
This algorithm comes from [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
2. Prioritized Batch Data
To optimize one-step update without losing diversity, we divide batch data into several parts and prioritize these batches. Using the batch data with maximal loss to do one-step update.
3. Distributed Multi-Agent Architecture
Similar to A3C algorithm, we adopt this Master and Multi-Worker architecture in our work.

## Experiment
### Implementation
- Keras 2.1.2 （tensorflow 1.4 as backend）
- mpi4py
- Python 3.6
- CUDA 8.0 + cuDNN 6.0

### Environment
- Modified original environment (you can find in my repo) from [OpenAI](https://github.com/openai/multiagent-particle-envs)
	- Fixed landmark
	- Border
<p align="center">
  <img width="300" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/env.png">
</p>

### Neural Network
<p align="center">
  <img width="800" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/network.PNG">
</p>

### Result 
<p align="center">
  <img width="600" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/result_curve.png">
</p>
<p align="center">
  <img width="400" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/result_table.PNG">
</p>

### Learning Progress
- DDPG & MADDPG & PROPOSED
<p align="center">
  <span width="280" style="float:left">
    <img width="280" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/ddpg_slow_gif.gif">
  </span>
  <span width="280" style="float:left">
    <img width="280" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/maddpg_slow_gif.gif">
  </span>
  <span width="280" style="float:left">
    <img width="280" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/proposed_slow_gif.gif">
  </span>
</p>

## How to run this program
For program using MPI:
- mpiexec -np [worker_number] python mpi-xxx.py
```python
mpiexec -np 4 python mpirun_main.py
```
For others:
```python
python xxx.py
```

## Future Work (4 vs 2)
<p align="center">
  <img width="300" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/4vs2_slow_gif.gif">
</p>

## Thanks to
- [MADDPG implementation repo](https://github.com/agakshat/maddpg)
- [OpenAI baselines](https://github.com/openai/baselines)
- [OpenAI envs](https://github.com/openai/multiagent-particle-envs)