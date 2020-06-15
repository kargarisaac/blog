---
title: "RL Series - DDPG"
description: "In this post, I talk about DDPG algorithm which is an off-policy RL algorithm for continous action spaces."
layout: post
toc: true
comments: true
hide: false
search_exclude: true
categories: [fastpages]
---

# RL Series - DDPG

This is part of my [RL-series](https://medium.com/@kargarisaac/rl-series-implementation-in-pytorch-bbeedb033866) posts.

This algorithm is from the _“Continuous Control with Deep Reinforcement Learning”_ [paper](https://arxiv.org/pdf/1509.02971.pdf) and uses the ideas from deep q-learning in the continuous action domain and is a model-free method based on the deterministic policy gradient.

In Deterministic Policy Gradient (DPG), for each state, we have one clearly defined action to take (the output of policy is one value for action and for exploration we add a noise, normal noise for example, to the action). But in Stochastic Gradient Descent, we have a distribution over actions (the output of policy is mean and variance of a normal distribution) and sample from that distribution to get the action, for exploration. In another term, in stochastic policy gradient, we have a distribution with mean and variance and we draw a sample from that as an action. When we reduce the variance to zero, the policy will be deterministic.

When the action space is discrete, such as q-learning, we get the max over q-values of all actions and select the best action. But in continuous action spaces, you cannot apply q-learning directly, because in continuous spaces finding the greedy policy requires optimization of a_t at every time-step and would be too slow for large networks and continuous action spaces. Based on the proposed equation in the reference paper, here we approximate max Q(s, a) over actions with Q(a, µ(s)).

In DDPG, they used function approximators, neural nets, for both action-value function Q and deterministic policy function µ. In addition, DDPG uses some techniques for stabilizing training, such as updating the target networks using soft updating for both μ and Q. It also uses batch normalization layers, noise for exploration, and a replay buffer to break temporal correlations.

This algorithm is an actor-critic method and the network structure is as follows:


![]({{ site.baseurl }}/images/posts_images/ddpg_post/ddpg_diagram.jpg "DDPG diagram")


First, the policy network gets the state and outputs the action mean vector. This will be a vector of mean values for different actions. For example, in a self-driving car, there are two continuous actions: steering and acceleration&braking (one continuous value between -x to x, the negative values are for braking and positive values are for acceleration). So we will have two mean for these two actions. To consider exploration, we can use Ornstein-Uhlenbeck or normal noise and add it to the action mean vector in the training phase. In the test phase, we can use the mean vector directly without any added noise. Then this action vector will be concatenated with observation and fed into the Q network. The output of the Q network will be one single value as a state-action value. In DQN, because it had discrete action space, we had multiple state-action values for each action, but here because the action space is continuous, we feed the actions into the Q network and get one single value as the state-action value.

Finally, the sudo code for DDPG is as follows:

![]({{ site.baseurl }}/images/posts_images/ddpg_post/ddpg_algorithm.jpg "DDPG algorithm")

To understand the algorithm better, it’s good to try to implement it and play with its parameters and test it in different environments. Here is a good implementation in PyTorch that you can start with [this](https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb). 

I also found the Spinningup implementation of DDPG very clear and understandable too. You can find it [here](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py)

For POMDP problems, it is possible to use LSTMs or any other RNN layers to get a sequence of observations. It needs a different type of replay buffer which I will write another post about that.
