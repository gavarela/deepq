## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Reinforcement Learning Routine
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' 

Depict game we want to play as a black box. It has a given state, you give it an action and it returns you a reward that depends on both the action and the state. The state changes and you repeat until the game ends (terminal state).

Idea is that the network predicts the Q-value of each action given a state. The Q-value is a sort of value/utility of the action (a) given the state (s):
    Q(s, a) = Sum_t y^t * r_t = r(s, a) + y * max_a'{Q(s', a')}
where r(s, a) is the immediate reward of action a in state s (Bellman equation).

We can thus pick actions by running the network on a given state and choosing the action with highest Q(s, a).

While playing game, we update the training examples and retrain the network to optimise the prediction. We add some random behaviour in order to allow the player to 'explore' the game (otherwise it repeats the actions it thinks are optimal over and over and can get stuck in a local minimum). The idea is that this iteration should make the network's estimate of Q(s, a) near the true Q(s, a).

I modified traditional Q-learning here by implementing a routine with a main 'master' network/player and a set of supplemental 'player' networks/players. The player networks are initialised to match the master network and play a series of games in parallel, training on each game they play after it ends. After a set number of games are played in parallel this way, the master network trains on their combined memoriesa and the process is repeated (starting from reinitialising the player networks). The idea behind this is not only that it should speed up training by playing games in parallel but that by having the player networks brach out, the games will be qualitatively different, allowing more of the game to be explored and then used to train the master network.

'''

import types, random, numpy as np, time

import multiprocessing as mp
from pathos.multiprocessing import ProcessPool

class RLRoutine(object):
    
    def __init__(self, cgame_class, player_class,
                 epsilon, disc_rate, max_memory_len,
                 use_keras = True, conv_net = True, 
                 l_rate = None, momentum = None,
                 filename = None):
        ''' Initialises QPlayer and its network and stores parameters. '''
        
        # Parameters
        if not isinstance(cgame_class, type):
            raise TypeError('Argument cgame_class must be a class (not an instance).')
        
        if not isinstance(player_class, type):
            raise TypeError('Argument player_class must be a class (not an instance).')
        
        self.cgame_class = cgame_class
        self.player_class = player_class
        self.epsilon = epsilon          # Function of epoch
        
        # Player
        self.master = self.player_class(disc_rate, max_memory_len, 'Master')
        if filename is None:
            
            in_shape, n_outputs = self.cgame_class.get_inp_out_dims(for_keras = use_keras, twod = conv_net)
            
            self.master.start_network(in_shape, n_outputs,
                                      use_keras, conv_net,
                                      l_rate, momentum)
        else:
            self.master.load_network(filename)
        
    def play_cgames(self, player, num_games, epsilon,
                    train = True,
                    l_rate = None, reg_rate = None, mom_rate = None):
        ''' Plays \num_games games with provided QPlayer. Always chooses action based on provided epsilon.
            Returs tuple containing QPlayer (memory), ComputerGame instance (history) and number of turns played. '''
        
        cgames = []
        turnlist = []
        
        for game in range(num_games):
            
            cgame = self.cgame_class()

            # First turn - remove d4
            ''' This is the only game (RemainOne) specific piece of code. Change eventually once things are working. '''
            action = self.cgame_class.get_inp_out_dims()[1]//2
            cgame.first_turn(action)

            new_state = cgame.get_state(for_keras = player.using_keras,
                                        twod = player.conv_net)
            
            #print(player.id + ' played first turn of game ' + str(game) + '.')
            
            # Loop over next turns
            turns = 0
            while not cgame.is_done():

                # Get action
                state = new_state
                legal_moves = cgame.legal_moves()

                action = player.get_move(
                            state,
                            random.random() < epsilon,
                            legal_moves,
                            cgame = cgame
                         )

                # Play turn
                reward = cgame.turn(action)
                new_state = cgame.get_state(for_keras = player.using_keras,
                                            twod = player.conv_net)

                # Store in temp memory
                player.store(state, action, reward, new_state, cgame.done, legal_moves)

                turns += 1
                #print(player.id + ' played turn ' + str(turns) + ' of game ' + str(game) + '.')
                
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
    
    def init_players(self, n_players):
        ''' Initialises list of qplayers identical to the main one. '''
        
        self.players = []
        for i in range(n_players):
            
            self.players.append(self.player_class(self.master.disc_rate,
                                                  self.master.memory.max_len,
                                                  'Player '+str(i)))
            
            self.players[-1].network = self.master.net_copy()
            self.players[-1].set_net_funcs(self.master.using_keras)
            self.players[-1].n_outputs = self.master.n_outputs
            self.players[-1].conv_net = self.master.conv_net
            self.players[-1].using_keras = self.master.using_keras
    
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
            #if (epoch+1) % player_life == 0:
                
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
                print('\nDeterministic game, %i mins into training:' % ((verbose_time-start_time)/60), end = ' ')
                
                _, cgame, turns = self.play_cgames(self.master, 1, 0, False)
                #self.master.memory.clear()
                
                print('lasted %i turns' % turns[0])
                #print('average %i turns with longest run %i turns.' % (sum(turns)/len(turns), max(turns)))
                self.det_turn_list += turns
            
            # Save master network
            if (epoch+1) % save_every == 0:
                
                print('\nSaving...')
                self.master.net_save(savedir + '/master_epoch_' + str(epoch+1) + ('.h5' if self.master.using_keras else '.json'))
                
            epoch += 1
            
        # Save master network
        self.master.net_save(savedir + '/master_final' + ('.h5' if self.master.using_keras else '.json'))
        