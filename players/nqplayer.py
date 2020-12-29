## ~~~~~~~~~~~~~~
#  n-step QPlayer
## ~~~~~~~~~~~~~~

import sys
if "../" not in sys.path:
    sys.path.append("../")

import numpy as np, random
from collections import deque

from utils.WillyNet import WillyNet
from players.qplayer import QPlayer


class nQPlayer(QPlayer):
    
    def train(self, mem_batch, batch_prop, 
              l_rate, reg_rate, mom_rate):
        
        self.trains += 1
        
        # Get batch
        if mem_batch < len(self.memory):
            batch = np.array(random.sample(self.memory, mem_batch))
        else:
            batch = np.array(self.memory)
        
        # Build examples and targets
        state, action, Reward, legal_moves = \
            [np.array(list(batch[:, i])).reshape((len(batch), -1)) for i in range(len(batch[0]))]
        
        examples = state
        
        targets = self.network.predict(state)
        targets *= legal_moves
        
        action_inds = np.tile(range(targets.shape[1]), (targets.shape[0], 1)) == action
        
        targets += action_inds * (-targets + Reward)
        
        # Train
        self.network.train(np.array(examples),
                           np.array(targets),
                           num_iterations = int(1/batch_prop),
                           batch_size = int(batch_prop*mem_batch),
                           learn_rate = l_rate,
                           reg_rate = reg_rate,
                           mom_rate = mom_rate)
    
    def play(self, game, epsilon, rand_start = False):
        
        game.restart(rand_start)
    
        hist = ''
        turns = 0
        temp_mem = []
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
            temp_mem.append((state, action, reward, lms))

            # History
            hist += str(action) + ' '
        
        R = 0
        for s, a, r, lms in reversed(temp_mem):
            R = r + self.disc_rate*R
            self.store(s, a, R, lms)

        return game._TOT_PIECES-1 - turns, hist