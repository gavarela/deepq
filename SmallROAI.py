## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Train AI to play Remain One using the 
#  Reinforcement Learning Routine
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np, multiprocessing as mp, time
import matplotlib.pyplot as plt
from utils import hist

import game.RemainOne as RO
import game.SmallRO as SRO
from rlroutine import RLRoutine
from players import qplayer as QP, vplayer as VP

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from utils.DW import *

#import willyai
#from willyai import DeepWilly
#from willyai.willies import ConvolutionalWilly, PoolingWilly, StackingWilly, ConnectedWilly, DropoutWilly

print('\033[33m')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\n          Small RO AI         \n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\033[0m')

## Initialise RL Routine
## ~~~~~~~~~~~~~~~~~~~~~

# Training parameters
param_cgame_class = SRO.ComputerGame
param_player_class = VP.VPlayer

param_epsilon = lambda x: 1.15 - 1/(1 + np.exp(-(x/25 - 25)))
param_disc_rate = 1
param_max_memory_len = 1500

param_epochs = 1000
param_n_players = mp.cpu_count() - 2 # <- 6
param_player_life = 3
param_train_players = True

param_batch_size = 50
param_l_rate = 0.000005
param_mom_rate = 0.85
param_reg_rate = 0.0005

param_use_keras = False
param_conv_net = True

param_verbose = 5         # Play deterministic game every x epochs
param_filename = None

param_save_every = 100
param_savedir = 'saves'

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
RL = RLRoutine(cgame_class = param_cgame_class,
               player_class = param_player_class,
               epsilon = param_epsilon,
               disc_rate = param_disc_rate,
               max_memory_len = param_max_memory_len,
               use_keras = param_use_keras,
               conv_net = param_conv_net,
               l_rate = param_l_rate,
               momentum = param_mom_rate)


## Train network to identify legal moves
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\033[32m\n')
print('Pre-Training Master Network')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\033[0m')

# Class to check legal moves
class MoveChecker(SRO.ComputerGame):
    ''' Uses legal_moves() method, as well as a new one, to set the state as we want it. '''

    def set_state(self, state, twod = False):
        ''' Sets state (board) of game to the given state, using a 2d or 1d description of the state. '''

        self.board = [[SRO.WALL]*8 for _ in range(8)]

        if twod:
            
            for r, row in enumerate(state):
                for c, slot in enumerate(row):
                    self.board[r+2][c+2] = slot

        else:

            i = 0
            for r in SRO.ROW_RANGE:
                for c in SRO.COL_RANGE[r]:
                    self.board[r][c] = state[i]
                    i += 1

# Generate dataset
print('Making dataset of states and moves now.')

in_dim, out_dim = param_cgame_class.get_inp_out_dims(param_conv_net, param_use_keras)

print('  With in and out dims:', in_dim, out_dim)

if param_conv_net:
    
    #shape = tuple([10000] + [s for s in in_dim])
    shape = (10000, 4, 4)
    states = np.random.choice([1, 0], shape, p = [1/3, 2/3])
    
    if param_cgame_class.__module__ == "game.RemainOne":
        for state in states:
            for inds in [(0, 0), (0, 5), (5, 0), (5, 5)]:
                state[inds[0]  , inds[1]  ] = 2
                state[inds[0]  , inds[1]+1] = 2
                state[inds[0]+1, inds[1]  ] = 2
                state[inds[0]+1, inds[1]+1] = 2
                
    elif param_cgame_class.__module__ == "game.SmallRO":
        for state in states:
            for inds in [(0, 0), (0, 3), (3, 0), (3, 3)]:
                state[inds[0], inds[1]] = 2

else:

    states = np.random.choice([1, 0], (10000, in_dim))

mc = MoveChecker()

moves = []
l = len(states)
for i, state in enumerate(states):
    if (i+1)%1000 == 0: print(' - Getting legal moves for state', i+1, 'of', l)
    mc.set_state(state, twod = param_conv_net)
    moves.append(mc.legal_moves())
moves = np.array(moves)

# Get rewards from num pieces in states and moves
if param_conv_net:
    n_pieces = (states==1).sum(2).sum(1).reshape((-1, 1))
else:
    n_pieces = (states==1).sum(1).reshape((-1, 1))

if param_player_class == VP.VPlayer:
    rewards = moves.max(axis = 1)[:, None]
    denom = n_pieces - (n_pieces != 1).astype(int)
    rewards = rewards / denom
else:
    rewards = 1/(n_pieces - 1*(n_pieces != 1)) * moves

if param_conv_net and param_use_keras:
    states = states.reshape((10000, 4, 4, 1))
elif param_conv_net:
    states = states.reshape((10000, 1, 4, 4))
else:
    states = states.reshape((10000, -1))

trainX = states[:9000]
trainy = rewards[:9000]

testX = states[9000:]
testy = rewards[9000:]

# Diagnostics
num_legal = np.sum(moves, axis = 1).reshape(-1, 1)

print('\nA proportion', np.mean((1 * num_legal>0)), 'of the training sample has any valid moves. \nHistogram:')

hist.hist(num_legal, 15, 'Examples in sample with x legal moves')

print('\nIn our sample, mean number of valid moves is %0.1f and max is %0.1f.' % (np.mean(num_legal), np.max(num_legal)))

print('\nMean reward from legal moves: %0.3f,\nMean reward from all moves: %0.3f,\nMax and min rewards: %0.3f and %0.3f.' %(np.mean(moves.max(1)), np.mean(moves), np.max(moves), np.min(moves)))

# Train
print('\nTraining master network now...')
start_time = time.time()

RL.master.net_train(trainX, trainy,
                    num_iterations = 250,
                    batch_size = 70,
                    l_rate = 0.005,
                    mom_rate = param_mom_rate,
                    reg_rate = param_reg_rate)

end_time = time.time()
print('  Done in', str(end_time - start_time) + ' secs.' if end_time - start_time < 60 else str((end_time - start_time)/60) + ' mins.')

cost = RL.master.net_cost(testX, testy)
print('  RMSE on test set:', (cost)**0.5)


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

print('Trained. Saving...')
filename = 'trained_RO_player'
filename += '.h5' if param_use_keras else '.json'
RL.master.net_save(filename)

# Save
file = open('last_run.json', 'w')
data = {'turn_list': RL.turn_list,
        'det_turn_list': RL.det_turn_list}
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




