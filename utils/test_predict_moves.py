## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Predict valid moves from state
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from RemainOne import *
import hist
import numpy as np, time

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import DW as DW
from DW import ConnectedWilly, ConvolutionalWilly, DropoutWilly, StackingWilly

param_twod = True


## Class to get data
## ~~~~~~~~~~~~~~~~~
class MoveChecker(ComputerGame):
    ''' Uses legal_moves() method, as well as a new one, to set the state as we want it. '''
    
    def set_state(self, state, twod = False):
        ''' Sets state (board) of game to the given state, using a 2d or 1d description of the state. '''
        
        if twod:
            
            for r, row in enumerate(state):
                for c, slot in enumerate(row):
                    self.board[r+2][c+2] = slot
            
            for inds in [(2, 2), (2, 7), (7, 2), (7, 7)]:
                self.board[inds[0]][inds[1]]     = WALL
                self.board[inds[0]][inds[1]+1]   = WALL
                self.board[inds[0]+1][inds[1]]   = WALL
                self.board[inds[0]+1][inds[1]+1] = WALL
        
                
        else:
            
            i = 0
            for r in ROW_RANGE:
                for c in COL_RANGE[r]:
                    self.board[r][c] = state[i]
                    i += 1
    

## Get dataset of states and moves
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('\nMaking dataset now.')

if param_twod:
    
    states = np.random.choice([FREE, FULL], (10000, 7, 7))
    for state in states:
        for inds in [(0, 0), (0, 5), (5, 0), (5, 5)]:
            state[inds[0]  , inds[1]  ] = 0
            state[inds[0]  , inds[1]+1] = 0
            state[inds[0]+1, inds[1]  ] = 0
            state[inds[0]+1, inds[1]+1] = 0

else:
    
    states = np.random.randint([FREE, FULL], (10000, 33))

mc = MoveChecker()
moves = []
l = len(states)
for i, state in enumerate(states):
    if (i+1)%1000 == 0: print(' - Getting legal moves for state', i+1, 'of', l)
    mc.set_state(state, twod = param_twod)
    moves.append(mc.legal_moves())

moves = np.array(moves)
states = states.reshape((10000, 7, 7, 1))

# Diagnostics
num_legal = np.sum(moves, axis = 1).reshape(-1, 1)

print('\nA proportion', np.mean((1 * num_legal>0)), 'of the training sample has any valid moves. \nHistogram:')

hist.hist(num_legal, 15, 'Examples in sample with x legal moves')

print('\nIn our sample, mean number of valid moves is:', np.mean(num_legal), 'and max is', np.max(num_legal))

moves = 1/(states.sum(3).sum(2).sum(1).reshape((-1, 1)) - 1) * moves

print('\nMean reward from legal moves: %0.3f,\nMean reward from all moves: %0.3f,\nMax and min rewards: %0.3f and %0.3f.' %(np.mean(moves.max(1)), np.mean(moves), np.max(moves), np.min(moves)))


## Build and run (C)NN
## ~~~~~~~~~~~~~~~~~~~

paramk_epochs = 100
paramk_bsize = 70

paramk_lrate = 0.08
paramk_momrate = 0.85


param_epochs = 200
param_bsize = 70

param_lrate = 0.05
param_momrate = 0.85
param_regrate = 0.1


## Build

# Build keras
net = Sequential()

net.add(Conv2D(20, kernel_size = (3, 3), activation = 'relu', input_shape = (7, 7, 1)))
#net.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
#net.add(MaxPooling2D(pool_size = (2, 2)))
#net.add(Dropout(0.25))
net.add(Flatten())
net.add(Dense(100, activation='relu'))
net.add(Dropout(0.2))
#net.add(Dense(128, activation='relu'))
#net.add(Dropout(0.5))
#net.add(Dense(moves.shape[1], activation='sigmoid'))
net.add(Dense(moves.shape[1], activation='linear'))

#net.compile(loss = keras.losses.categorical_crossentropy,
#            optimizer = keras.optimizers.SGD(lr = paramk_lrate,
#                                              momentum = paramk_momrate),
#                #keras.optimizers.Adadelta(),
#            metrics = ['categorical_accuracy'])

# Build DeepWilly
dnet = DW.DeepWilly(cost_func = 'cross entropy')

dnet.add(ConvolutionalWilly(20, (3, 3), in_shape = (1, 7, 7)))
dnet.add(StackingWilly())
dnet.add(ConnectedWilly(100))
dnet.add(DropoutWilly(0.2))
dnet.add(ConnectedWilly(moves.shape[1], act_func = 'sigmoid'))


## Train and evaluate

# 3 times with diff test and train sets
ktimes, kaccs = [], []
dwtimes, dwaccs = [], []
for i in range(3):
    
    print('Doing iteration', i+1, '/', 3)
    
    # Re-do test and train sets
    shuffled_inds = np.random.permutation(range(10000))
    
    testX = states[shuffled_inds][9000:]
    testy = moves[shuffled_inds][9000:]

    trainX = states[shuffled_inds][:9000]
    trainy = moves[shuffled_inds][:9000]

    # keras
    
    net.compile(loss = keras.losses.mean_squared_error, 
                optimizer = keras.optimizers.SGD(lr = paramk_lrate, 
                                                 momentum = paramk_momrate),
                metrics = [keras.losses.mean_squared_error])
        # ^Do that to reinitialise weights
    
    print('  Training keras...')
    start = time.time()
    net.fit(trainX, trainy,
            batch_size = paramk_bsize,
            epochs = paramk_epochs,
            verbose = 0,# if i != 2 else 2,
            validation_data = (testX, testy))
    end = time.time()
    print('    Trained keras in', end - start, 's.')

#    my_score = np.mean((net.predict(testX) > 0.5) == testy)
#    print('    Keras test accuracy:', my_score)
    
    score = net.evaluate(testX, testy)[0]
    print('    Keras MSE:', score, ', RMSE:', score**0.5)
    
    ktimes.append(end-start)
#    kaccs.append(my_score)
    kaccs.append(score)

    # DW
#    print('  Training DeepWilly...')
#
#    start = time.time()
#    dnet.train(X = trainX.reshape((-1, 1, 7, 7)), 
#               y = trainy,
#               learn_rate = param_lrate,
#               batch_size = param_bsize,
#               reg_rate = param_regrate,
#               num_iterations = param_epochs,
#               mom_rate = param_momrate,
#               verbose = None)# if i!= 2 else param_epochs // 10)
#    end = time.time()
#    print('    Trained DeepWilly in', end - start, 's.')
#
#    # Test
#    acc = dnet.accuracy(testX.reshape((-1, 1, 7, 7)), 
#                        testy, pred_type = 'binary')
#    print('    DW accuracy:', acc)
#    dwtimes.append(end-start)
#    dwaccs.append(acc)

# Print
print('\n Results: keras    ~    DeepWilly',
      '\n\nParams:',
      '\n - epochs    :', paramk_epochs, '   ~   ', param_epochs,
      '\n - batch size:', paramk_bsize, '   ~   ', param_bsize,
      '\n - learn rate:', paramk_lrate, '   ~   ', param_lrate,
      '\n - mom rate  :', paramk_momrate, '   ~   ', param_momrate,
      '\n - reg rate  :', '   ', '   ~   ', param_regrate,
      '\n\nMean time:', np.mean(ktimes), '   ~   ', np.mean(dwtimes),
        '\nMean acc :', np.mean(kaccs), '   ~   ', np.mean(dwaccs))

net.save('keras_net.h5')

# End
print()