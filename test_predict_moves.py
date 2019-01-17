## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Test - NN to predict valid moves
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import NeuralNet as NN
from RemainOne import *
import numpy as np
import hist

## New class
## ~~~~~~~~~

class MoveChecker(ComputerGame):
    
    def set_state(self, state):
        
#        print('\nStarting set_state()...')
#        print(self.places)
#        
        for i, val in enumerate(state):
            place = self.places[i]
#            print(place)
            if val:
                if not place.is_full():
#                    print('\nGoing to add piece to place:')
#                    print(place)
                    Piece(self, place)
            else:
                if place.is_full():
#                    print('\nGoing to remove piece from place:')
#                    print(place)
                    place.remove_piece()
    
    def legal_moves(self):
        ''' Returns list with indices corresponding to moves that are legal. '''
        
        legal_moves = []
        for move in self.computer_moves:
            legal_moves.append(int(move.is_valid()))
        
        return legal_moves
    

## Get dataset of states and moves
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\nMaking dataset now.')

states = np.random.randint(0, 2, (10000, 33))

mc = MoveChecker()
moves = []
l = len(states)
for i, state in enumerate(states):
    if (i+1)%1000 == 0: print(' - Getting legal moves for state', i+1, 'of', l)
    mc.set_state(state)
    moves.append(mc.legal_moves())
moves = np.array(moves)

num_legal = np.sum(moves, axis = 1).reshape(-1, 1)

print('\nA proportion', np.mean((1 * num_legal>0)), 'of the training sample has any valid moves. \nHistogram:')

hist.hist(num_legal, 15, 'Examples in sample with x legal moves')

print('\nIn our sample, mean number of valid moves is:', np.mean(num_legal), 'and max is', np.max(num_legal))

testX = states[:9000]
testy = moves[:9000]

trainX = states[9000:]
trainy = moves[9000:]


## Build NN
## ~~~~~~~~

print('\nDone. Training network...')

# Parameters
n_in = trainX.shape[1]
n_out = trainy.shape[1]

param_shape = [n_in, 200, 200, n_out]
param_cost = NN.CrossEntropyCost
param_act = NN.Sigmoid
param_weights = 'XHWeights'

param_epochs = 200
param_bsize = 70
param_ltype = 'SGD'
param_lrate = 0.15
param_momrate = None
param_regtype = 'L2'
param_regrate = 0.12

param_verbose = 20

# Train network
net = NN.NeuralNet(sizes = param_shape, 
                   act_func = param_act, 
                   cost_func = param_cost, 
                   weights = param_weights)

net.train(X = trainX, y = trainy, 
          epochs = param_epochs, 
          batch_size = param_bsize, 
          eta = param_lrate, 
          mu = param_momrate, 
          method = param_ltype, 
          lmbda = param_regrate, 
          reg = param_regtype, 
          verbose = param_verbose)

# Test
Yhat = net.predict(testX, 'binary')
acc = net.accuracy(testy)

print('\nWith parameters: ')
print(' - Shape:                   ', param_shape)
print(' - Weights:                 ', param_weights)
print(' - Cost:                    ', param_cost.__name__)
print(' - Activation:              ', param_act.__name__)
print(' - Epochs:                  ', param_epochs)
print(' - Batch size:              ', param_bsize)
print(' - Learning Type:           ', param_ltype)
print(' - Learning rate:           ', param_lrate)
print(' - Momentum friction:       ', param_momrate)
print(' - Regularisation:          ', param_regtype)
print(' - Regularisation parameter:', param_regrate)

print('\nPrediction accuracy:', acc, '\n')

# Save network
net.save('valid_moves_net.json')

