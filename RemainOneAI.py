## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Apply the deep Q reinforcement learning
#  code to the Remain One game
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np

import RemainOne as RO
import NeuralNet as NN
import DeepQLearning as DQL

np.random.seed(82736)

## Apply RL routine
## ~~~~~~~~~~~~~~~~

n_in, n_out = RO.ComputerGame.get_inp_out_dims()

# Parameters
param_sizes = [n_in, 200, 200, n_out]
param_act_func = NN.Sigmoid
param_cost_func = NN.CrossEntropyCost

param_epsilon = lambda g: 1.15 - 1/(1 + np.exp(-(g/100 - 22))) 
                             # 1 if t > g//250 + 1 else 0.15
param_disc_rate = 0.7
param_max_memory_len = 4000
param_memory_props = {1: 1/2, -10: 1/2}

param_games = 10000
param_batch_size = lambda x: int(max(1.5*x, 150))
param_learn_rate = 0.12
param_reg = None # 'L2'
param_reg_rate = 0 # 0.04

param_verbose = False

# RL routine
RL = DQL.RLRoutine(param_epsilon, param_disc_rate, param_max_memory_len,
               param_sizes, param_act_func, param_cost_func, filename = 'valid_moves_net.json')

RL.learn(RO.ComputerGame, 
         param_games, param_batch_size, param_learn_rate,
         param_memory_props,
         param_reg, param_reg_rate,
         param_verbose)

print('Trained.')
print(RL.qplayer.network.weights)
input('Press Enter to see the AI play.')

RL.play_new(RO.HumanGame)