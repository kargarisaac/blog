---
title: "RL algorithms"
description: "In this post, I talk about DDPG algorithm which is an off-policy RL algorithm for continous action spaces."
layout: post
toc: true
comments: true
hide: false
search_exclude: true
categories: [RL]
---

# RL Algorithms

In this post I will overview different single and multi-agent Reinforcement Learning (RL) algorithms. I will update this post and add algorithms periodically.

![RL diagram]({{ site.baseurl }}/images/posts_images/rl-series/rl-diagram.png "RL diagram")

Here are some resources to learn more about RL!

- David Silver's [course](https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)

- CS287 at UC Berkeley - Advanced Robotics [course](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNBPJdt8WamRAt4XKc639wF) - Instructor: Pieter Abbeel

- CS 285 at UC Berkeley - Deep Reinforcement Learning [course](http://rail.eecs.berkeley.edu/deeprlcourse/) - Instructor: Sergey Levine

- CS234 at Stanford - Reinforcement Learning [course](http://web.stanford.edu/class/cs234/index.html) - Instructor: Emma Brunskill

- CS885 at University of Waterloo - Reinforcement Learning [course](https://www.youtube.com/playlist?list=PLdAoL1zKcqTXFJniO3Tqqn6xMBBL07EDc) - Instructor: Pascal Poupart 

- Arthur Juliani's [posts](https://medium.com/@awjuliani)

- Jonathan Hui's [posts](https://medium.com/@jonathan_hui/rl-deep-reinforcement-learning-series-833319a95530)

- A Free [course](https://simoninithomas.github.io/deep-rl-course/) in Deep Reinforcement Learning from beginner to expert by Thomas Simonni

## Single agent

### DQN

We will take a look at DQN with experience replay buffer and the target network. 

DQN is a value-based method. It means that we try to learn a value function and then use it to achieve the policy. In DQN we use a neural network as a function approximator for our value function. It gets the state as input and outputs the value for different actions in that state. These values are not limited to be between zero and one, like probabilities, and can have other values based on the environment and the reward function we define.

DQN is an off-policy method which means that we are using data from old policies, the data that we gather in every interaction with the environment and save it in the experience replay buffer, to sample from it later and train the network. The size of the replay buffer should be large enough to reduce the i.i.d property between data that we sample from it.

To use DQN, the action should be discrete. We can use it for continuous action spaces by discretizing the action space, but it’s better to use other techniques that can handle continuous action spaces such as Policy Gradients.
First, let’s see the algorithm’s sudo code:

![DQN algorithm]({{ site.baseurl }}/images/posts_images/rl-series/dqn.png "DQN algorithm")

In this algorithm, we have experience replay buffer and a target network with a different set of parameters that will be updated every C steps. These tricks help to get a better and more stable method rather than pure DQN. There are a lot of improvements for DQN and we will see some of them in the next posts too.

First, we initialize the weights of both networks and then start from the initial state s and take action a with epsilon-greedy policy. In the epsilon-greedy policy, we select an action a randomly or using the Q-network. Then we execute the selected action and get the next state, reward, and the done values from the environment and save them in our replay buffer. Then we sample a random batch from the replay buffer and calculate target based on the Bellman equation in the above picture and use MSE loss and gradient descent to update the network weights. We will update the weights of our target network every C steps.

In the training procedure, we use epsilon decay. It means that we consider a big value for epsilon, such as 1. Then during the training procedure, as we go forward, we reduce its value to something like 0.02 or 0.05, based on the environment. It will help the agent to do more exploration in the first steps and learn more about the environment. It’s better to have some exploration always. That’s a trade-off between exploration-exploitation.
In test time, we have to use a greedy policy. It means we have to select the action with the highest value, not randomly anymore (set epsilon to zero actually).


### Reinforce

### A2C

### A3C

### PPO

### DDPG

This algorithm is from the _“Continuous Control with Deep Reinforcement Learning”_ [paper](https://arxiv.org/pdf/1509.02971.pdf) and uses the ideas from deep q-learning in the continuous action domain and is a model-free method based on the deterministic policy gradient.

In Deterministic Policy Gradient (DPG), for each state, we have one clearly defined action to take (the output of policy is one value for action and for exploration we add a noise, normal noise for example, to the action). But in Stochastic Gradient Descent, we have a distribution over actions (the output of policy is mean and variance of a normal distribution) and sample from that distribution to get the action, for exploration. In another term, in stochastic policy gradient, we have a distribution with mean and variance and we draw a sample from that as an action. When we reduce the variance to zero, the policy will be deterministic.

When the action space is discrete, such as q-learning, we get the max over q-values of all actions and select the best action. But in continuous action spaces, you cannot apply q-learning directly, because in continuous spaces finding the greedy policy requires optimization of $a_t$ at every time-step and would be too slow for large networks and continuous action spaces. Based on the proposed equation in the reference paper, here we approximate _max Q(s, a)_ over actions with _Q(a, µ(s))_.

In DDPG, they used function approximators, neural nets, for both action-value function $Q$ and deterministic policy function $\mu$. In addition, DDPG uses some techniques for stabilizing training, such as updating the target networks using soft updating for both $\mu$ and $Q$. It also uses batch normalization layers, noise for exploration, and a replay buffer to break temporal correlations.

This algorithm is an actor-critic method and the network structure is as follows:


![DDPG diagram]({{ site.baseurl }}/images/posts_images/ddpg_post/ddpg_diagram.jpg "DDPG diagram")


First, the policy network gets the state and outputs the action mean vector. This will be a vector of mean values for different actions. For example, in a self-driving car, there are two continuous actions: steering and acceleration&braking (one continuous value between $-x$ to $x$, the negative values are for braking and positive values are for acceleration). So we will have two mean for these two actions. To consider exploration, we can use Ornstein-Uhlenbeck or normal noise and add it to the action mean vector in the training phase. In the test phase, we can use the mean vector directly without any added noise. Then this action vector will be concatenated with observation and fed into the $Q$ network. The output of the $Q$ network will be one single value as a state-action value. In DQN, because it had discrete action space, we had multiple state-action values for each action, but here because the action space is continuous, we feed the actions into the $Q$ network and get one single value as the state-action value.

Finally, the sudo code for DDPG is as follows:

![DDPG algorithm]({{ site.baseurl }}/images/posts_images/ddpg_post/ddpg_algorithm.jpg "DDPG algorithm")

To understand the algorithm better, it’s good to try to implement it and play with its parameters and test it in different environments. Here is a good implementation in PyTorch that you can start with [this](https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb). 

I also found the Spinningup implementation of DDPG very clear and understandable too. You can find it [here](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py)

For POMDP problems, it is possible to use LSTMs or any other RNN layers to get a sequence of observations. It needs a different type of replay buffer for sequential data.

### SAC

### Ape-X

### R2D2

### IMPALA

### Never Give-Up

### Agent57

## Multi-Agent 

### MADDPG

### COMA
