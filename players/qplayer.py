## ~~~~~~~~~~~~~~~~~
#  Q-learning player
## ~~~~~~~~~~~~~~~~~

import numpy as np
from .memory import *

#import keras
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from utils.DW import *
from utils.WillyNet import *

#import willyai
#from willyai import DeepWilly
#from willyai.willies import ConvolutionalWilly, PoolingWilly, StackingWilly, ConnectedWilly, DropoutWilly


class QPlayer(object):
    ''' Will have a memory to store experiences and a network (brain) to choose moves/actions with. Network is a CNN. '''
    
    def __init__(self, disc_rate, max_memory_len, identifier):
        
        self.memory = Memory(max_memory_len)
        self.disc_rate = disc_rate
        
        self.id = identifier
        
    def start_network(self, in_shape, n_outputs,
                      use_keras = True, conv_net = True, 
                      l_rate = None, momentum = None):
        ''' Initialises a convolutional neural network. Uses keras if use_keras = True, uses DeepWilly if not. '''
        
        self.n_outputs = n_outputs
        self.using_keras = use_keras
        self.conv_net = conv_net
        
        if not conv_net:
            
            self.network = WillyNet([in_shape, 200, 200, n_outputs], 
                                    problem = 'regression')
            
            self.set_net_funcs(use_keras = False)
        
        elif use_keras:
            
            if l_rate is None or momentum is None:
                raise ValueError("Need to provide learning rate and momentum when initialising keras network!")
            
            self.network = Sequential()
            self.network.add(Conv2D(20, kernel_size = (3, 3), input_shape = in_shape, activation = 'relu'))
            self.network.add(Flatten())
            self.network.add(Dense(100, activation = 'relu'))
            self.network.add(Dropout(0.2))
            self.network.add(Dense(n_outputs, activation = 'linear'))

            self.network.compile(
                loss = keras.losses.mean_squared_error, 
                optimizer = keras.optimizers.SGD(lr = l_rate, momentum = momentum),
                metrics = [keras.losses.mean_squared_error])
            
            self.set_net_funcs(use_keras = True)
            
        else:
            
            self.network = DeepWilly(cost_func = 'quadratic')
            
            self.network.add(ConvolutionalWilly(20, (3, 3), in_shape = in_shape))
            #self.network.add(ConvolutionalWilly(20, (2, 2)))
            self.network.add(StackingWilly())
            self.network.add(DropoutWilly(0.2))
            self.network.add(ConnectedWilly(200))
            self.network.add(DropoutWilly(0.1))
            self.network.add(ConnectedWilly(100))
            self.network.add(ConnectedWilly(n_outputs, act_func = 'linear'))
            
            self.set_net_funcs(use_keras = False)
        
    def load_network(self, filename, use_keras = True):
        
        self.using_keras = use_keras
        
        if use_keras:
            self.network = keras.models.load_model(filename)
            self.set_net_funcs(use_keras = True)
        else:
            self.network = DeepWilly.load(filename)
            self.set_net_funcs(use_keras = False)
    
    def copy_keras(self):
        
        copy = keras.models.clone_model(self.network)
        copy.compile(
            loss = keras.losses.mean_squared_error, 
            optimizer = keras.optimizers.SGD(lr = self.network.optimizer.lr, 
                                             momentum = self.network.optimizer.momentum),
            metrics = [keras.losses.mean_squared_error] \
                    )
        
        copy.set_weights(self.network.get_weights())
        return copy
    
    def set_net_funcs(self, use_keras):
        ''' Define network-dependent class functions for forward propagation (net_predict), training (net_train) and cost evaluation (net_cost). '''
        
        if use_keras:
            
            self.net_train = lambda trainX, trainy, \
                                    num_iterations, batch_size, \
                                    l_rate, mom_rate, reg_rate: \
                self.network.fit(trainX, trainy,
                                 batch_size = batch_size,
                                 epochs = num_iterations,
                                 verbose = 0)
                
            
            self.net_predict = self.network.predict
            
            self.net_cost = lambda testX, testy: \
                self.network.evaluate(testX, testy)[0]
            
            self.net_copy = self.copy_keras
            
            self.net_save = self.network.save
            
        else:
            
            self.net_train = lambda trainX, trainy, \
                                    num_iterations, batch_size, \
                                    l_rate, mom_rate, reg_rate: \
                self.network.train(X = trainX, y = trainy,
                                   num_iterations = num_iterations,
                                   batch_size = batch_size,
                                   learn_rate = l_rate,
                                   reg_rate = reg_rate,
                                   mom_rate = mom_rate)
            
            self.net_predict = self.network.predict
            
            self.net_cost = lambda testX, testy: \
                self.network.cost(self.net_predict(testX), testy)
            
            self.net_copy = self.network.copy
            
            self.net_save = self.network.save
    
    def get_move(self, state, random_state, legal_moves):
        ''' Selects move with highest predicted Q-value given the state from legal moves. If random_state is True, chooses randomly with predicted Q-values as the probability distribution.'''
        
        # Q-values for each move
        preds = self.net_predict(np.array([state]))
        move_vals = preds * legal_moves
        move_vals *= (move_vals > 0)
        move_vals += legal_moves * 1.e-99 
        
        if np.sum(np.isnan(move_vals)) > 0:
            print('\n\nNAN IN MOVE_VALS for ' + self.id + '!!!:\n',
                  move_vals)
            input('PE.')
        
        if np.sum(legal_moves) == 0:
            print('\n\nNO LEGAL MOVES for ' + self.id + '!!!:\n',
                  legal+moves)
            input('PE.')
        
        # Random
        if random_state:
            
#            if np.sum(move_vals) == 0:
#                print('\n\nSUM MOVE VALS = 0 for ' + self.id + '!!!:\npreds, legal moves, move vals:\n',
#                      preds, '\n', legal_moves, '\n', move_vals)
#                input('PE.')
#            
#            probs = move_vals / np.sum(move_vals)
#            
#            return np.random.choice(range(move_vals.shape[1]),
#                                    p = probs.ravel())
            
            return np.random.choice(np.where(legal_moves == 1)[0])
        
        # Deterministic
        else:
            
            return np.argmax(move_vals)
            
        
    def store(self, state, action, reward, new_state, crash, legal_moves):
        
        self.memory.append([state, action, reward, new_state, crash, legal_moves], 
                           multiple = False)
        
    def retrieve(self, ind = None):
        
        if ind is None: return self.memory.memory
        return self.memory[ind]
    
    def train(self, train_set_size, batch_size, l_rate, 
              reg_rate = 0.01, mom_rate = 0.85,
              use_last = False, verbose = False):
        ''' Trains the network on a batch drawn randomly from memory. If use_last == True, uses the last \batch_size items in memory. 
           Prepares batch, rowwise, by:
            - running network on s' and on s;
            - use as target the result from running on s with the following corrections:
              a) Q-value corresponding to a -> r + y * max_a'{Q(s', a')};
              b) Q-value corresponding to illegal moves -> 0.
            '''
        
        # Gen batch for training
        if train_set_size == -1:
            train_set = np.array(self.memory.memory)
        elif use_last:
            train_set = np.array(self.memory.memory[-min(len(self.memory), train_set_size):])
        else:
            train_set = np.random.permutation(self.memory)[:train_set_size]
        
        if train_set_size == -1:
            train_set_size = len(train_set)
        else:
            train_set_size = min(len(train_set), train_set_size)
        
        # Training examples
        examples = np.array(list(train_set[:, 0]))
        
        # Update targets for training examples in batch
        targets = self.net_predict(examples)
        if verbose: old_fp = np.copy(targets)
        
        for u, unit in enumerate(train_set):
            
            # Inds: 0 - state, 1 - action, 2 - reward, 3 - new state, 4 - crash, 5 - legal moves
            
            new_pred = self.net_predict(np.array([unit[0]]))
            max_new_pred = np.max(new_pred)
            
            targets[u, unit[1]] = unit[2] + \
                (not unit[4]) * self.disc_rate * max_new_pred
            
            legal_moves = np.array([1 if i in unit[5] else 0 for i in range(self.n_outputs)])
            targets[u] *= legal_moves
            
        # Given examples and targets, train
        self.net_train(examples, targets, 
                       num_iterations = 1, 
                       batch_size = batch_size,
                       l_rate = l_rate,
                       mom_rate = mom_rate,
                       reg_rate = reg_rate)
        
        # Print stuff
        if verbose:
            
            new_fp = self.net_predict(examples)
            
            print('  RMSE:', np.mean((old_fp - targets)**2)**0.5, 
                  '->', np.mean((new_fp - targets)**2)**0.5,
                  '\n  Cost:', self.network.cost(old_fp, targets),
                        '->', self.network.cost(new_fp, targets))
    
    
    