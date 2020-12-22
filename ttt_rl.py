## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Reinforcement Learning Routine
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import types, random, numpy as np, time
from rlroutine import *
from players.qplayer import *

import multiprocessing as mp
from pathos.multiprocessing import ProcessPool


## Random Player
## ~~~~~~~~~~~~~

class RandPlayer(QPlayer):
    
    def __init__(self, max_memory_len): 
        
        self.memory = Memory(max_memory_len)
    
    def start_network(self, in_shape, n_outputs,
                      use_keras = True, conv_net = True, 
                      l_rate = None, momentum = None): None
    
    def load_network(self, filename, use_keras = True): None
    
    def get_move(self, state, random_state, legal_moves):
        
        return np.random.choice(np.where(legal_moves == 1)[0])
    
    def train(self, train_set_size, batch_size, l_rate, 
              reg_rate = 0.01, mom_rate = 0.85,
              use_last = False, verbose = False): None
    
    def set_net_funcs(self, use_keras):
        
        self.net_train = lambda trainX, trainy, num_iterations, batch_size, l_rate, mom_rate, reg_rate: None
    
        self.net_predict = lambda x: None
            
        self.net_cost = lambda testX, testy: None

        self.net_copy = None

        self.net_save = None



## RL Routine
## ~~~~~~~~~~

class TTT_RL(RLRoutine):
    
    def play_cgames(self, player, num_games, epsilon,
                    train = True,
                    l_rate = None, reg_rate = None, mom_rate = None):
        ''' Plays \num_games games with provided QPlayer. Always chooses action based on provided epsilon.
            Returs tuple containing QPlayer (memory), ComputerGame instance (history) and number of turns played. '''
        
        cgames = []
        turnlist = []
        
        for game in range(num_games):
            
            cgame = self.cgame_class()

            # Loop over turns
            new_state = cgame.get_state(for_keras = player.using_keras, twod = player.conv_net)
            turns = 0
            while not cgame.is_done():

                # Get action
                state = new_state
                legal_moves = cgame.legal_moves()

                action = player.get_move(
                            state,
                            random.random() < epsilon,
                            legal_moves
                         )

                # Play turn
                reward = cgame.turn(action)
                new_state = cgame.get_state(for_keras = player.using_keras, twod = player.conv_net)

                # Store in temp memory
                player.store(state, action, reward, new_state, cgame.done, legal_moves)

                turns += 1
                
            # Train on game that just ended
            if train:
                player.train(train_set_size = turns,
                             batch_size = -1,
                             l_rate = l_rate,
                             reg_rate = reg_rate,
                             mom_rate = mom_rate,
                             use_last = True)
            
            cgames.append(cgame)
            turnlist.append(turns)
            
        return player, cgames, turnlist
    
    def test_games(self, num_games):
        
        rand_agent = RandPlayer(self.master.memory.max_len)
        
        winlist = []
        losslist = []
        turnlist = []
        for game in range(num_games):
            
            cgame = self.cgame_class()

            # Loop over next turns
            new_state = cgame.get_state(for_keras = self.master.using_keras, twod = self.master.conv_net)
            play = random.choice([0, 1])
            turns = 0
            win = 0
            loss = 0
            while not cgame.is_done():
                
                turn_player = self.master if play % 2 == 0 else rand_agent
                
                # Get action
                state = new_state
                legal_moves = cgame.legal_moves()

                action = turn_player.get_move(
                            state,
                            random.random() < -1,
                            legal_moves
                         )

                # Play turn
                reward = cgame.turn(action)
                new_state = cgame.get_state(for_keras = self.master.using_keras, twod = self.master.conv_net)

                # Store in master's memory
                self.master.store(state, action, reward, new_state, cgame.done, legal_moves)
                
                # End iteration
                if play % 2 == 0:
                    win += reward
                else:
                    loss += reward
                play += 1
                turns += 1
                
            # Append number of turns
            turnlist.append(turns)
            winlist.append(win)
            losslist.append(loss)
            
        return turnlist, winlist, losslist
    
    def learn(self, epochs, batch_size, 
              player_life, train_players = True,
              l_rate = None, reg_rate = None, mom_rate = None,
              verbose = False, n_players = None,
              save_every = None, savedir = None):
        ''' Whole RL routine for learning. In a loop:
            1. Runs many games in parallel, adding to memory but not training; 
            2. Train on previous set of games played in parallel;
            3. Every set number of iterations, re-initiate all player networks to match master network;
            3. Every other set number of iterations, play a game with no randomness to track progress. 
            '''
        
        # Handle args: \n_players and \verbose
        n_players = n_players or max(mp.cpu_count() - 2, 1)
        verbose = verbose or np.Inf
        save_every = save_every or np.Inf
        
        # Play games
        self.cgame_list = []
        self.turn_list = []
        
        self.det_turn_list = []
        self.det_win_list = []
        self.det_loss_list = []
        
        self.init_players(n_players)
        pool = ProcessPool(n_players)
        
        print('Beginning learning process. Will run for a total of  %i epochs, running %i players at a time, each with a life of %i epochs.' % (epochs, n_players, player_life))
        
        start_time = time.time()
        
        epoch = 0
        while epoch < epochs:
            
            print('\nDoing epoch', epoch+1, 'of', epochs, end = ':\n')
            
            # Play \n_players games in parallel
            def play_cgames(player):
                return self.play_cgames(player, player_life, 
                                        self.epsilon(epoch), train_players,
                                        l_rate, reg_rate, mom_rate)
            output = pool.map(play_cgames, self.players)
            
            self.players = []
            for o in output:
                self.players.append(o[0])
                self.cgame_list += o[1]
                self.turn_list += o[2]
            
            print('  Mean turns played: %0.1f, max: %0.1f, memories of total length %i ' %(np.mean([o[2] for o in output]), np.max([o[2] for o in output]), sum([len(o[0].memory) for o in output])))
            
            # Initialise new player networks and train master network
            for player in self.players:
                self.master.memory.append(player.memory.memory)

            self.init_players(n_players)

            self.master.train(train_set_size = -1,
                              batch_size = batch_size,
                              l_rate = l_rate,
                              reg_rate = reg_rate,
                              mom_rate = mom_rate,
                              verbose = True)

            self.master.memory.clear()
        
            # Print progress
            if (epoch+1) % verbose == 0:
                
                verbose_time = time.time()
                print('\n10 deterministic games, %i mins into training:' % ((verbose_time-start_time)/60), end = ' ')
                
                turns, wins, losses = self.test_games(10)
                
                print(':\n  mean turns: %0.1f\n  max turns: %i\n  number wins: %i\n  number losses: %i' % (np.mean(turns), max(turns), sum(wins), sum(losses)))
                #print('average %0.1f turns with longest run %i turns.' % (sum(turns)/len(turns), max(turns)))
                self.det_turn_list += turns
                self.det_win_list += wins
                self.det_loss_list += losses
            
            # Save master network
            if (epoch+1) % save_every == 0:
                
                print('\nSaving...')
                self.master.net_save(savedir + '/master_epoch_' + str(epoch+1) + ('.h5' if self.master.using_keras else '.json'))
                
            epoch += 1
            
        # Save master network
        self.master.net_save(savedir + '/master_final' + ('.h5' if self.master.using_keras else '.json'))
        