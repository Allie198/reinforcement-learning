# Reinforcement Learning Projects

This repository contains a collection of reinforcement learning implementations
built using **PyTorch** and **Gymnasium**, focusing on both **classic control**
and **high-dimensional environments**.

The goal of this repo is to provide **clean, understandable, and extensible**
implementations of modern RL algorithms, with an emphasis on **theory-to-code
consistency** rather than black-box usage.

---

## Implemented Algorithms & Environments

### Deep Convolutional Q-Network (DCQN) – Atari Breakout

**Environment:** `ALE/Breakout-v5`  
**Algorithm:** DCQN (DQN with CNN-based state encoder)

- Raw pixel input processing with convolutional neural networks
- Experience replay buffer
- Target network for stabilized learning
- ε-greedy exploration strategy
- Frame stacking and reward clipping
 
---

### Advantage Actor-Critic (A2C) – CartPole

**Environment:** `CartPole-v1`  
**Algorithm:** A2C

- Shared backbone for actor and critic
- On-policy learning
- Advantage-based policy gradient updates
- Entropy regularization for exploration

 

---

### Proximal Policy Optimization (PPO) – LunarLander

**Environment:** `LunarLander-v2`  
**Algorithm:** PPO (Clipped Objective)

- Clipped surrogate loss for stable updates
- Generalized Advantage Estimation (GAE)
- Separate actor and critic networks
- Mini-batch optimization over multiple epochs
 

---

### Soft Actor-Critic (SAC) – Humanoid

**Environment:** `Humanoid-v5`  
**Algorithm:** SAC

- Maximum entropy reinforcement learning
- Twin Q-networks to reduce overestimation bias
- Automatic entropy coefficient tuning
- Continuous action space control
