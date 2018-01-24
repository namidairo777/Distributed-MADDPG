# Distributed-MADDPG
Master Course Graduation Project - Distributed Multi-Agent Cooperation Algorithm based on MADDPG with prioritized batch data.
## Introduction
This work focus on Multi-Agent Cooperation Problem. We proposed a method which consists 3 components:
1. Related research - MADDPG
This algorithm comes from [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
2. Prioritized Batch Data
To optimize one-step update without losing diversity, we divide batch data into several parts and prioritize these batches. Using the batch data with maximal loss to do one-step update.
3. Distributed Multi-Agent Architecture
Similar to A3C algorithm, we adopt this Master and Multi-Worker architecture in our work.

## Distributed Multi-Agent Architecture
![architecture](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/architecture.png)

## Experiment
### Environment
- Modified original environment (you can find in my repo) from [OpenAI](https://github.com/openai/multiagent-particle-envs)
	- Fixed landmark
	- Border
![env](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/env.png)

### Neural Network

![network](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/network.png)

### Result 
![result_curve](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/result_curve.png)
![result_table](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/result_table.png)

### Learning Progress
![ddpg_slow_gif](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/ddpg_slow_gif.gif)
![maddpg_slow_gif](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/maddpg_slow_gif.gif)
![proposed_slow_gif](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/proposed_slow_gif.gif)

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
![4vs2_slow_gif](https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/4vs2_slow_gif.gif)
<p align="center">
  <img width="460" height="300" src="https://github.com/namidairo777/Distributed-MADDPG/blob/master/imgs/4vs2_slow_gif.gif">
</p>