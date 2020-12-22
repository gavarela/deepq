## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Train AI to play Tic-Tac-Toe using the 
#  Reinforcement Learning Routine
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np, multiprocessing as mp, time
import matplotlib.pyplot as plt
from utils import hist

import game.TTT as TTT
from ttt_rl import *
from players import qplayer as QP, vplayer as VP

from utils.DW import *

print('\033[33m')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\n         Tic Tac Toe AI         \n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\033[0m')

## Initialise RL Routine
## ~~~~~~~~~~~~~~~~~~~~~

# Training parameters
param_cgame_class = TTT.ComputerGame
param_player_class = QP.QPlayer

relu = lambda z: z * (z > 0)
param_epsilon = lambda x: relu(1 - 0.02*x)
param_disc_rate = 1
param_max_memory_len = 1500

param_epochs = 750
param_n_players = mp.cpu_count() - 2 # <- 6
param_player_life = 3
param_train_players = True

param_batch_size = 50
param_l_rate = 0.000005
param_mom_rate = 0.85
param_reg_rate = 0.0005

param_use_keras = False
param_conv_net = False

param_verbose = 5         # Play deterministic game every x epochs
param_filename = None

param_save_every = 100
param_savedir = 'ttt_saves'

print('\033[32m\n')
print('Parameters')
print('~~~~~~~~~~')
print('\033[0m')

print('Using %s in %s:' % (param_player_class.__name__,
                           param_cgame_class.__module__),
      '\n - discount rate       =', param_disc_rate,
      '\n - max memory len      =', param_max_memory_len, '\n',
      '\n - epochs              =', param_epochs,
      '\n - num players       =', param_n_players,
      '\n - player life         =', param_player_life, '\n',
      '\n - batch size          =', param_batch_size,
      '\n - learning rate       =', param_l_rate,
      '\n - momentum rate       =', param_mom_rate,
      '\n - regularisation rate =', param_reg_rate, '\n',
      '\n - using keras         =', param_use_keras,
      '\n - convolutional net   =', param_conv_net)

# RL Routine
RL = TTT_RL(cgame_class = param_cgame_class,
               player_class = param_player_class,
               epsilon = param_epsilon,
               disc_rate = param_disc_rate,
               max_memory_len = param_max_memory_len,
               use_keras = param_use_keras,
               conv_net = param_conv_net,
               l_rate = param_l_rate,
               momentum = param_mom_rate)


## Start reinforcement learning
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\033[32m\n')
print('Reinforcement Learning Routine')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\033[0m')

RL.learn(epochs = param_epochs,
         batch_size = param_batch_size,
         player_life = param_player_life,
         train_players = param_train_players,
         l_rate = param_l_rate,
         reg_rate = param_reg_rate,
         mom_rate = param_mom_rate,
         verbose = param_verbose,
         n_players = param_n_players,
         save_every = param_save_every,
         savedir = param_savedir)

print('Trained and saved.')

# Save
file = open(param_savedir + '/last_run.json', 'w')
data = {'turn_list': RL.turn_list,
        'det_turn_list': RL.det_turn_list,
        'det_win_list': RL.det_win_list,
        'det_loss_list': RL.det_loss_list}
json.dump(data, file)
file.close()

# Plot performance over time
epoch_means = np.array(RL.turn_list).reshape((-1, param_n_players)).mean(axis = 1)

fig, ax = plt.subplots()

ax.plot(range(len(epoch_means)), epoch_means, 'k.', markersize = 1, label = 'Training games')
ax.plot(np.arange(len(RL.det_turn_list))*param_verbose, RL.det_turn_list, 'r-', markersize = 1, label = 'Testing games')

ax.set_title('AI Performance')
ax.set_xlabel('Epoch')
ax.set_ylabel('Number of Turns')

plt.show()




