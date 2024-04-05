# Intermediate Reinforcement Learning Project - Ornithopter 

Welcome to the Intermediate Project of the Reinforcement Learning Course by the Ornithopter group, in which we tackled three different problems using Reinforcement Learning algorithms.

## Our Contributors
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/RicardoRibeiroRodrigues"><img src="https://avatars.githubusercontent.com/RicardoRibeiroRodrigues" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Ricardo Ribeiro Rodrigues</b></sub></a><br />Developer</td>
    <td align="center"><a href="https://github.com/Pedro2712"><img src="https://avatars.githubusercontent.com/Pedro2712" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Pedro Andrade</b></sub></a><br />Developer</td>
    <td align="center"><a href="https://github.com/JorasOliveira"><img src="https://avatars.githubusercontent.com/JorasOliveira" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Jorás Oliveira</b></sub></a><br />Developer</td>
    <td align="center"><a href="https://github.com/renatex333"><img src="https://avatars.githubusercontent.com/renatex333" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Renato Laffranchi</b></sub></a><br />Developer</td>
  </tr>
</table>
</div>

## Overview
In this project, we engage with the following environments:

* Lunar Lander from [Farama Foundation Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/);

<p align="center">
  <img src="https://gymnasium.farama.org/_images/lunar_lander.gif" alt="Lunar Lander Example Gif">
</p>

* Cart Pole from [Farama Foundation Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/);

<p align="center">
  <img src="https://gymnasium.farama.org/_images/cart_pole.gif" alt="Cart Pole Example Gif">
</p>

* Flappy Bird implemented by [Martin Kubovčík](https://github.com/markub3327/flappy-bird-gymnasium).

<p align="center">
  <img src="https://raw.githubusercontent.com/markub3327/flappy-bird-gymnasium/main/imgs/dqn.gif" alt="Flappy Bird Example Gif" width="50%">
</p>

Our goal is to develop and evaluate the DQN and Double DQN algorithms on all environments listed above. Through iterative design and testing, our team seeks to optimize agents performance and compare learning efficiency.

## Algorithms

In this project, we have implemented both DQN and Double DQN algorithms. 

### DQN (Deep Q-Networks)

Something goes here.

### Double DQN

Something else goes here.

## Comparison: DQN vs Double DQN

By comparing these two algorithms, we not only gauged the impact of advanced techniques like experience replay on the learning efficiency and stability but also demonstrated the evolution of reinforcement learning strategies from basic Q-Learning to sophisticated architectures like DQN. The implementation of both algorithms provided valuable insights into the dynamics of reinforcement learning and its application in complex environments such as lunar landing.

### Learning Curves


![Learning Curve Lunar Lander Environment](results/results.jpg)

![Learning Curve Cart Pole Environment](results/results.jpg)

![Learning Curve Flappy Bird Environment](results/results.jpg)

The learning curve above demonstrates the agent's performance over time, measured in terms of average reward per episode. Initially, the agent struggles to achieve successful landings, often incurring penalties for crashes or excessive fuel consumption. Over time, as the agent learns from its experiences, we observe a positive trend in performance, with increased rewards indicating more successful and efficient landings.

### Agent Demonstrations

Here are two animations showing the agent trained using Deep Q-Learning algorithm in action:

- Initial Stages of Learning:

<div align="center">
  <p align="center">
  <img src="results/lander_trained_dql_half.pt.gif" alt="Initial Stages of Deep Q-Learning">
</p>
</div>

- After Training Completion:

<p align="center">
  <img src="results/lander_trained_dql.pt.gif" alt="Trained Deep Q-Learning Agent">
</p>

Here are two animations showing the agent trained using DQN algorithm in action:

- Initial Stages of Learning:
  
<p align="center">
  <img src="results/lander_trained_dqn_half.pt.gif" alt="Initial Stages of Deep Q-Networks">
</p>

- After Training Completion:

  <p align="center">
  <img src="results/lander_trained_dqn.pt.gif" alt="Trained Deep Q-Networks Agent">
</p>

These GIFs illustrate the progression from an inexperienced agent to a skilled one, capable of handling the complexities of lunar landing.

### Conclusion

It is clear that in both algorithms, the agents were able to learn and specialize in their tasks. However, the Deep Q-Network (DQN) outperformed Deep Q-Learning (DQL) since, in the comparison of the Reward vs. Episode curves, the DQN converged in far fewer episodes, sometimes even in half the number of episodes required by DQL, and also achieved a higher average reward than DQL.

The reason the Deep Q-Network (DQN) might perform better than basic Deep Q-Learning could be due to a single main improvements in DQN. DQN uses what's called a target network, which is a separate network that helps in making the learning process more stable. This feature helps DQN learn faster and achieve better results compared to traditional Deep Q-Learning.

## References

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Hw222fiZ)