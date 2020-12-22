## ~~~~~~~~~~~~~~~~~~
#  nQ-learning player
## ~~~~~~~~~~~~~~~~~~

'''

n-step Q-learning agent.

'''

import numpy as np, math
from .qplayer import *

class nQPlayer(QPlayer):
    ''' Will only need to change init, store and train. See nVPlayer code to see the parallels. '''
    
    def __init__(self, disc_rate, max_memory_len, identifier):
        
        super().__init__(self, disc_rate, math.inf, identifier)
        self.max_memory_len = max_memory_len
        
        self.game_len = 0
    
    def store(self, state, action, reward, new_state, crash, legal_moves):
        ''' No need to store future state'''
        
        self.memory.append([state, action, reward, crash, legal_moves])
        self.game_len += 1
        
        if crash is True:
            
            R = 0
            for i in range(1, self.game_len+1):
                R = self.memory[-i][2] + self.disc_rate * R
                self.memory[-i][2] = R
            
            self.game_len = 0
            
            overload = len(self.memory) - self.max_memory_len
            if overload > 0:
                self.memory.memory = self.memory.memory[overload:]
            
    def train(self, train_set_size, batch_size, l_rate, 
              reg_rate = 0.01, mom_rate = 0.85,
              use_last = False, verbose = False):
        ''' Prepares batch to train on by:
             - running network on s;
             - use as target the result from running on s with the following corrections:
               a) Q-value corresponding to a -> R;
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
            
            # Inds: 0 - state, 1 - action, 2 - reward, 3 - crash, 4 - legal moves
            
            targets[u, unit[1]] = unit[2]
            
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