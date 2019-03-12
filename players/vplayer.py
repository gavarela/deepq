## ~~~~~~~~~~~~~~~~~
#  V-learning player
## ~~~~~~~~~~~~~~~~~

''' 

A new proposed reinforcement learning paradigm, which I think may be useful in games in which (like RemainOne):

1. Many actions may at some point be valid (76 in this game) but in any given state, only a small subset of those (4 or so) will actually be valid;
2. Given the state, each valid action will lead to a distinct future state;
3. Given the state, each valid action provides the same return.

The proposition is: instead of having the player's network try to estimate the Q-value of each action in the given state (i.e. 76 Q(s, a), where most of those 76 should be 0), let's have it estimate V(s) = max_a Q(s, a) and chose actions by calculating the V(s') for each future state s' which is achievable given today's state (s)'s valid actions.

The smaller number of quantities to estimate (and the fact that there used to be sparse, mostly 0) may be more tractable for the network.

Let's give it a shot!

'''

import numpy as np
from .memory import *
from .qplayer import *

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from utils.DW import *
from utils.WillyNet import *


class VPlayer(QPlayer):
    
    def start_network(self, in_shape, n_outputs,
                      use_keras = False, conv_net = True,
                      l_rate = None, momentum = None):
        ''' Initialises a conv net by calling QPlayer's start_network with n_outputs set to 1: just need to estimate value of a state, not the Q-value of each action in that state. '''
        
        super().start_network(in_shape, 1,
                              use_keras, conv_net,
                              l_rate, momentum)
        
    def get_move(self, state, random_state, legal_moves,
                **kwargs):
        ''' Selects move from legal_moves whose future state has the highest predicted value. If random_state is true, chooses randomly with the predicted values as the probability distribution. '''
        
        # Get attainable future states
        legal_moves = np.where(legal_moves)[0]
        
        next_state_values = []
        for move in legal_moves:
            next_state = kwargs['cgame'].next_state(move, twod = self.conv_net, for_keras = self.using_keras)
            next_value = self.net_predict(np.array([next_state]))
            next_state_values.append(next_value)
        
        # Random
        if random_state:
            
#            probs = np.array(next_state_values)
#            probs *= probs > 0
#            probs /= np.sum(probs)
            
            return np.random.choice(legal_moves)#,
#                                    p = probs.ravel())
        
        # Deterministic
        else:
            
            return legal_moves[np.argmax(next_state_values)]
        
    def train(self, train_set_size, batch_size, l_rate, 
              reg_rate = 0.01, mom_rate = 0.85,
              use_last = False, verbose = False):
        ''' Trains the network on a batch drawn randomly from memory. If use_last == True, uses the last \batch_size items in memory. 
           Prepares batch, rowwise, by:
            - running network on s';
            - use (r + y*V(s')) as target (or 0 if crashed).
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
        
        # Training examples and targets
        examples = np.array(list(train_set[:, 0]))
        
        rewards = np.array(list(train_set[:, 2]))[:, None]
        next_states = np.array(list(train_set[:, 3]))
        crashed = np.array(list(train_set[:, 4]))[:, None]
        
        targets = (rewards + self.disc_rate * self.net_predict(next_states)) * crashed
        
        #print('\nIn VPlayer train, shapes of rewards, next_states, predicted_vals, crashed and targets:', rewards.shape, next_states.shape, self.net_predict(next_states).shape, crashed.shape, targets.shape)
        
        # Train
        if verbose: old_fp = self.net_predict(examples)
        
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
        
        
        
        
        
    
    