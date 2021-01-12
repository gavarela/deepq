## ~~~~~~~~~~
#  Q-Player
## ~~~~~~~~~~

import sys
if "../" not in sys.path:
    sys.path.append("../")

import numpy as np, random
from collections import deque
from itertools import islice

from utils.WillyNet import WillyNet


class QPlayer(object):
    
    def __init__(self, max_mem_len, disc_rate, shape_or_filename):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate

        self.memory = deque()
        self.long_term_mem = deque()

        if isinstance(shape_or_filename, list):
            
            self.network = WillyNet(shape = shape_or_filename, 
                                    problem = 'regression')
        
        else:
            
            self.network = WillyNet.load(shape_or_filename)
    
    def store(self, *args):
        
        if len(self.memory) + len(self.long_term_mem) == self.max_mem_len:
            self.memory.popleft()
        
        self.memory.append(list(args))
        
    def remember(self, mem_prop):
        
        self.long_term_mem += deque(islice(self.memory, 0, mem_prop*self.max_mem_len))
        
        for i in range(mem_prop*self.max_mem_len):
            self.memory.popleft()
    
    def get_action(self, state, at_random, legal_moves):
        
        lm_inds = np.where(legal_moves)[0]
        
        if at_random:
            ret = random.choice(lm_inds)
            return ret
        
        else:
            ret = self.network.predict(np.array([state]))
            ret = ret[0, lm_inds]
            return lm_inds[ret.argmax()]
    
    def train(self, mem_batch, batch_size, 
              l_rate, reg_rate, mom_rate):
        
        # Get batch
        if mem_batch < len(self.memory):
            batch = np.array(random.sample(self.memory+self.long_term_mem, mem_batch))
        else:
            batch = np.array(self.memory+self.long_term_mem)
        
        # Build examples and targets
        state, action, reward, new_state, crashed, legal_moves = \
            [np.array(list(batch[:, i])).reshape((len(batch), -1)) for i in range(len(batch[0]))]
        
        examples = state
        
        targets = self.network.predict(state)
        targets *= legal_moves
        
        action_inds = np.tile(range(targets.shape[1]), (targets.shape[0], 1)) == action
        
        targets += action_inds * (-targets + reward)
        
        next_Q = self.network.predict(new_state).max(axis = 1).reshape((len(batch), -1))

        targets += action_inds * (1-crashed) * self.disc_rate * next_Q
        
        # Train
        self.network.train(np.array(examples),
                           np.array(targets),
                           num_iterations = 1,
                           batch_size = batch_size,
                           learn_rate = l_rate,
                           reg_rate = reg_rate,
                           mom_rate = mom_rate)
    
    def play(self, game, epsilon, rand_start = False):
    
        game.restart(rand_start)

        hist = ''
        turns = 0
        new_state = game.get_state()
        while not (game.crashed or game.won):

            # Play turn
            state = new_state
            lms = game.lms

            action = self.get_action(state, 
                                     random.random() < epsilon,
                                     lms)

            reward = game.turn(action)
            new_state = game.get_state()

            turns += 1

            # Store in memory
            self.store(state, action, reward, new_state, game.crashed, lms)

            # History
            hist += str(action) + ' '

        return game._TOT_PIECES-1 - turns, hist