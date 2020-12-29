## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Policy Gradient Ascent Player
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
if "../" not in sys.path:
    sys.path.append("../")

import numpy as np, random
from collections import deque

from utils.WillyNet import WillyNet
from players.qplayer import QPlayer


def one_hot(A, n_classes):
    return np.eye(n_classes)[A]

class PolPlayer(QPlayer):
    
    def __init__(self, max_mem_len, disc_rate,
                 state_size, hidden_shape, n_actions):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate
        
        self.memory = deque()
        
        self.trains = 0
        
        # Network
        self.state_size = state_size
        self.n_actions = n_actions
        
        shape = [state_size] + hidden_shape + [n_actions]
        self.network = WillyNet(shape = shape,
                                problem = 'softmax')
    
    def get_action(self, state, at_random, legal_moves):
        
        lm_inds = np.where(legal_moves)[0]
        
        a_probs = self.network.predict(np.array([state]))
        
        legal_probs = a_probs[0, lm_inds]
        
        if at_random:
            return random.choice(lm_inds)
            #entropy = -sum(a_probs * np.log(a_probs))
            legal_probs /= sum(legal_probs)
            return np.random.choice(lm_inds, p = legal_probs)
        
        else:
            return lm_inds[legal_probs.argmax()]
    
    def train(self, mem_batch, batch_prop, 
              l_rate, reg_rate, mom_rate):
        
        self.trains += 1
        
        # Get batch
        if mem_batch < len(self.memory):
            batch = np.array(random.sample(self.memory, mem_batch))
        else:
            batch = np.array(self.memory)
        
        # Build examples and targets
        state, action, Reward = \
            [np.array(list(batch[:, i])).reshape((len(batch), -1)) for i in range(len(batch[0]))]
        
        a_probs = self.network.predict(state)
        a_taken = one_hot(action.reshape(-1), self.n_actions)
        
        X = state
        y = a_taken
        W = -np.tile(Reward, self.n_actions)
            # minus for gradient ascent on policy^
        
        # Train
        self.network.train(X, y,
                           weights = W,
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
            temp_mem.append((state, action, reward))
            
            # History
            hist += str(action) + ' '
        
        R = 0
        for s, a, r in reversed(temp_mem):
            R = r + self.disc_rate*R
            self.store(s, a, R)

        return _TOT_PIECES-1 - turns, hist