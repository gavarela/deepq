## ~~~~~~~~~~
#  A2C Player
## ~~~~~~~~~~

import sys
if "../" not in sys.path:
    sys.path.append("../")

import numpy as np, random
from collections import deque

from utils.WillyNet import WillyNet
from players.qplayer import QPlayer


## A2C Neural Network
## ~~~~~~~~~~~~~~~~~~

class A2CNet(WillyNet):
    ''' Modify WillyNet in order to build an A2C network that shares all layers but for the output one. Only diff is return from predict and softmax on all but one output.. '''
    
    softmax = lambda z: np.exp(z)/(np.sum(np.exp(z), axis = 1).reshape((-1, 1)) + 1e-99)
    
    def __init__(self, state_size, hidden_shape, n_actions, 
                 hidden_act = 'relu', weights = 'XavierHe'):
        
        shape = [state_size] + hidden_shape + [n_actions + 1]
        super().__init__(shape, 'regression', hidden_act, weights)
        
    def forward_prop(self, states):
        ''' Returns values and action probabilities for each state. '''
        
        super().forward_prop(states)
        
        print('Pre softmax: ', self.A[-1][0])
        action_probs = A2CNet.softmax(self.A[-1][:, :-1])
        print('Post softmax:', action_probs[0])
        values = self.A[-1][:, -1].reshape((-1, 1))
        print('Value:', values[0], '\n')
        
        self.A[-1] = np.hstack([action_probs, values])
        
        return action_probs, values
        
    def predict(self, states, choose_act = False):
        ''' Returns values and actions (or action probabilities) for each state. '''
        
        states = np.array(states)
        acts, vals = self.forward_prop(states)
        
        if not choose_act:
            return acts, vals
        else:
            chosen_acts = acts.argmax(axis = 1).reshape((-1, 1))
            return chosen_acts, vals


## A2C Player
## ~~~~~~~~~~

def one_hot(A, n_classes):
    return np.eye(n_classes)[A]

class A2CPlayer(QPlayer):
    
    def __init__(self, max_mem_len, disc_rate,
                 state_size, hidden_shape, n_actions):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate
        
        self.memory = deque()
        
        self.trains = 0
        
        # Network
        self.state_size = state_size
        self.n_actions = n_actions
        
        self.network = A2CNet(state_size, hidden_shape, n_actions)
        
    def get_action(self, state, at_random, legal_moves):
        
        lm_inds = np.where(legal_moves)[0]
        
        a_probs, _ = self.network.predict(np.array([state]))
        
        legal_probs = a_probs[0, lm_inds]
        
        if at_random:
            return random.choice(lm_inds)
            
            #entropy = -sum(a_probs * np.log(a_probs))
            
            #print('legal_probs:', legal_probs)
            legal_probs /= sum(legal_probs)
            #print(' -> ', legal_probs)
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
        state, action, reward, next_state, crashed, lms = \
            [np.array(list(batch[:, i])).reshape((len(batch), -1)) for i in range(len(batch[0]))]
        
        a, V = self.network.predict(state)
        _, next_V = self.network.predict(next_state)
        
        a_taken = one_hot(action.reshape(-1), self.n_actions)
        
        R = reward + self.disc_rate * next_V * (1 - crashed)
        
        #print("s, a, r, s', c, lms:", state.shape, action.shape, reward.shape, next_state.shape, crashed.shape, lms.shape)
        #print('a, V, nV, at, R', a.shape, V.shape, next_V.shape, a_taken.shape, R.shape)
        #print('\na:', action[:2].reshape(-1))
        #print('\nV:', V[:2].reshape(-1))
        #print('\nR:', R[:2].reshape(-1), '\n')
        
        X = state
        y = np.hstack([a_taken, R])
        W = np.hstack([-np.tile(R - V, self.n_actions), np.ones_like(R)])
            # minus for gradient ascent on policy^
        
        # Train
        self.network.train(X, y, 
                           weights = W,
                           num_iterations = int(1/batch_prop),
                           batch_size = int(batch_prop*mem_batch),
                           learn_rate = l_rate,
                           reg_rate = reg_rate,
                           mom_rate = mom_rate)