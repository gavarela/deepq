## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Apply the deep Q reinforcement learning
#  code to the Remain One game
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np, multiprocessing as mp, time

import matplotlib.pyplot as plt, hist

import RemainOne as RO
import WillyNet as WN
import DeepQLearning as DQL

#np.random.seed(82736)

## Initialise RL routine
## ~~~~~~~~~~~~~~~~

n_in, n_out = RO.ComputerGame.get_inp_out_dims()

# Parameters
param_shape = [n_in, 200, 200, n_out]

param_epsilon = lambda x: 1.15 - 1/(1 + np.exp(-(x/40 - 22)))

param_disc_rate = 0.8
param_max_memory_len = 4000
param_memory_props = None

param_epochs = 2000
param_n_processes = mp.cpu_count() - 2
    # Will play param_epochs * param_n_processes games

param_batch_size = 150
param_l_rate = 0.01
param_reg_rate = 0
param_mom_rate = 0.9

param_verbose = 20 # Warn of progress every 20 epochs

param_filename = 'valid_moves_net.json'

# RL routine
RL = DQL.RLRoutine(param_epsilon, 
                   param_disc_rate,
                   param_max_memory_len, param_memory_props,
                   param_shape)


## Train network to identify legal moves
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Class to check which moves are legal given a state
class MoveChecker(RO.ComputerGame):
    
    def set_state(self, state):
        
        for i, val in enumerate(state):
            place = self.places[i]
            if val:
                if not place.is_full():
                    RO.Piece(self, place)
            else:
                if place.is_full():
                    place.remove_piece()
    
    def legal_moves(self):
        ''' Returns list with indices corresponding to moves that are legal. '''
        
        legal_moves = []
        for move in self.computer_moves:
            legal_moves.append(int(move.is_valid()))
        
        return legal_moves
    

# Get dataset of states and moves
print('\nMaking dataset now.')

states = np.random.randint(0, 2, (10000, 33))
n_states = len(set([tuple(state) for state in states]))
print('\nContains', n_states, 'distinct states', '(%0.1f%%)' % (n_states/len(states) * 100))

mc = MoveChecker()
moves = []
l = len(states)
for i, state in enumerate(states):
    if (i+1)%1000 == 0: print(' - Getting legal moves for state', i+1, 'of', l)
    mc.set_state(state)
    moves.append(mc.legal_moves())
moves = np.array(moves)

num_legal = np.sum(moves, axis = 1).reshape(-1, 1)

print('\n%0.1f%% of the training sample has any valid moves.' % (np.mean((1 * num_legal>0))*100) + \
      '\nMean number of valid moves is %0.1f and max is %0.1f' % (np.mean(num_legal), np.max(num_legal)))

print('Histogram:')
hist.hist(num_legal, 15, 'Examples in sample with x legal moves')

testX = states[:9000]
testy = moves[:9000]

trainX = states[9000:]
trainy = moves[9000:]

# Train NN on this and print accuracy
print('\nFitting network to data...', end = ' ')
start_fit = time.time()
try:
    
    #MLPRegressor
    
    RL.qplayer.network.fit(X = trainX, y = trainy)
    end_fit = time.time()
    print('Done in %i s.' %end_fit - start_fit)
    
    
    preds = RL.qplayer.network.predict(testX)
    
except:
    
    # WillyNet
    
    RL.qplayer.network.train(X = trainX,
                             y = trainy, 
                             l_rate = 0.15, 
                             batch_size = 70,
                             reg_rate = 0.12, 
                             num_iterations = 200,
                             verbose = False)
    
    end_fit = time.time()
    print('Done in %i s.' % (end_fit - start_fit))
    
    preds = RL.qplayer.network.forward_prop(testX)

print('RMSE on test set:', np.mean((preds - testy)**2)**0.5)


## Do rest of RL routine
## ~~~~~~~~~~~~~~~~~~~~~

print('\nOnto RL routine now.')

RL.learn(RO.ComputerGame, 
         param_epochs,
         param_batch_size,
         param_l_rate,
         param_reg_rate,
         param_mom_rate,
         param_verbose,
         param_n_processes)

print('Trained. Saving...')
RL.qplayer.network.save('trained_RO_player.json')

# Plot performance over time
rand_turns = np.mean(np.array(RL.turn_list).reshape((-1, param_n_processors)), 
                     axis = 1)

fig, ax = plt.subplots()

ax.plot(range(len(rand_turns)), RL.turnlist, 'k.', markersize = 1, label = 'Training games')
ax.plot(np.array(range(len(RL.det_turn_list)))*param_verbose, RL.det_turn_list, 'r-', label = 'Testing games')

ax.set_title('AI Performance')
ax.set_xlabel('Epoch')
ax.set_ylabel('Number of Turns')

plt.show()