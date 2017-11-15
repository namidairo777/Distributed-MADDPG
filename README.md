# Distributed-MADDPG
Master Course Graduation Project - Distributed Multi-Agent DDPG using PPO as policy optimizer
## Composition introduction
1. Actor network in ddpg
- input: state_dim 
- => dense(64) + ReLU + norm
- => dense(64) + ReLU + norm 
- => output: dense(action_dim) + softmax
```Python
Model(inputs=input_obs,outputs=pred)
model.compile(optimizer='Adam',loss='categorical_crossentropy')
```

## ToDO
A lot of work to do.