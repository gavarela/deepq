## ~~~~~~~~~~~~~~~~~~
#  Testing Q-Learning
#  Small Remain One
## ~~~~~~~~~~~~~~~~~~


'''

IF WANT TO CHANGE BOARD SIZE, FOLLOWING CHANGES NEED TO BE MADE:
 - BOARD, N_PIECES, ROW_RANGE and COL_RANGE
 - In Game __init__: self.board, self.n_pieces and self.do_first_move stuff
 - In print_board: Indexing (abc and 123)
 - In det game bit: Score appended to det_turnlist
 - SHAPE of network
 

MRO2 times:
playing game is ~ 0.002s
training is ~ 0.04s

'''

import sys
if "../" not in sys.path:
    sys.path.append("../")
    
#import numpy as np, multiprocessing as mp, random, time
#from pathos.multiprocessing import ProcessPool
#
from testMRO2_checkpoints import *
#
#
### Function to play many parallel games with one player
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#def restart(game, rand_start):
#    game.restart(rand_start)
#    return game
#
#def turn(game, action):
#    r = game.turn(action)
#    return game, r
#
#def play_cgames(pool, games, player, epsilon, rand_start = False):
#    
#    n_games = len(games)
#    
#    games = pool.map(restart, games, [rand_start for i in range(n_games)])
#    
#    new_states = np.array(pool.map(lambda g: g.get_state(),
#                                   games))
#    while not (sum([g.crashed or g.won for g in games]) == len(games))  :
#        
#        states = new_states
#        lms = np.array(pool.map(lambda g: g.lms,
#                                games))
#        
#        lm_inds = [np.where(l)[0] for l in lms]
#        rands = [random.random() < epsilon \
#                 for i in range(n_games)]
#        
#        actions = np.array([0 if games[i].crashed or games[i].won else \
#                            random.choice(lm_inds[i]) if rands[i] else \
#                            None \
#                            for i in range(n_games)])
#        
#        dets = [actions[i] is None \
#                for i in range(n_games)]
#        if sum(dets) > 0:
#            
#            pactions = player.network.predict(states[np.where(dets)[0], :])
#            
#            lmis = [lm_inds[i] for i in range(n_games) if dets[i]]
#            pactions = [lmis[i][pactions[i][lmis[i]].argmax()] \
#                        for i in range(sum(dets))]
#            
#            actions[np.where(dets)[0]] = pactions
#        
#        res = pool.map(turn, games, actions)
#        games = [r[0] for r in res]
#        rewards = [r[1] for r in res]
#        
#        new_states = np.array(pool.map(lambda g: g.get_state(),
#                                   games))
#        
#        for i in range(len(games)):
#            player.store(states[i], actions[i], rewards[i], 
#                         new_states[i], games[i].crashed,
#                         lms[i])
    


## Do RL
## ~~~~~

if __name__ == "__main__":
    
    import time, re, shutil
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n     Test - MRO2 parallelised     \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    
    ## Load params
    ## ~~~~~~~~~~~
    
    PLAYER_CLASS = QPlayer
    DIRNAME = 'testMRO_saves/MRO2/test3'
        # ^ DO NOT ADD '/' to end!
    
    # Get params
    with open(DIRNAME + '/params.json', 'r') as file:
        PARAMS = json.load(file)
    
    # Rename folder if set to
    if PARAMS['RENAME']:
        
        DIRNAME = DIRNAME.split('/')
        DIRNAME[-1] = 'e%ik g%i mb%i tb%i lr%s rr%s mr%s %srand' \
            %(PARAMS['NUM_EPOCHS']/1e3,
              PARAMS['MEM_BATCH'],
              PARAMS['TRAIN_BATCH'] * PARAMS['MEM_BATCH'],
              str(PARAMS['L_RATE']).replace('.', '-'),
              str(PARAMS['REG_RATE']).replace('.', '-'),
              str(PARAMS['MOM_RATE']).replace('.', '-'),
              '' if PARAMS['RAND_DETS'] else 'no')
        DIRNAME = '/'.join(DIRNAME)
        os.rename(DIRNAME, DIRNAME)
        
        PARAMS['RENAME'] = False
        with open(DIRNAME + '/params.json', 'w') as file:
            json.dump(PARAMS, file)
            
    else: 
        DIRNAME = DIRNAME
    
    # Setup stuff (anew or from latest checkpoint)
    if os.path.isdir(DIRNAME + '/checkpoints'):
        filenames = os.listdir(DIRNAME + '/checkpoints')
        resume = bool(len(filenames))
    else:
        os.mkdir(DIRNAME + '/checkpoints')
        resume = False
    
    if not resume:
        
        SHAPE = [N_PIECES] + \
                PARAMS['INNER_SHAPE'] + \
                [len(Game._moves)]
        player = PLAYER_CLASS(PARAMS['MAX_MEM_LEN'],
                              PARAMS['DISC_RATE'],
                              SHAPE)
        
        # For training
        turnlist, winlist = [], []
        det_turnlist, det_winlist = [], []
        time_base = 0
        
        CURRENT_EPOCH = 0
    
    else:
        
        # Get latest (current) epoch
        matches = [[m for m in re.finditer('_(\d+)\.', file)] for file in filenames]
        numbers = [int(match[0].group(1)) if len(match) > 0 else -1 for match in matches]
        CURRENT_EPOCH = max(numbers)

        # Get progress data
        file = open(DIRNAME + '/checkpoints/results_%i.json' %CURRENT_EPOCH, 'r')
        data = json.load(file)
        file.close()

        turnlist = data['turnlist']
        winlist = data['winlist']

        det_turnlist = data['det_turnlist']
        det_winlist = data['det_winlist']

        time_base = data['time']
        
        # Set up player
        player = PLAYER_CLASS(PARAMS['MAX_MEM_LEN'],
                          PARAMS['DISC_RATE'],
                          DIRNAME + '/checkpoints/network_%i.json' %CURRENT_EPOCH)
    
    # Final stuff
    EPSILON = lambda i: sum([PARAMS['EPSILON'][j] * i**j \
                             for j in range(len(PARAMS['EPSILON']))])
    
#    pool = ProcessPool(PARAMS['N_PAR_GAMES'])
    
    game  =  Game()
#    games = [Game() for i in range(PARAMS['N_PAR_GAMES'])]
    
    
    # Reinforcement learning
    # ~~~~~~~~~~~~~~~~~~~~~~
    
    last_hist = ''
    start_time = time.time()
    for epoch in range(CURRENT_EPOCH, PARAMS['NUM_EPOCHS']):
        
        # Play games
        n_games = 1 + \
             (PARAMS['GAMES_PER_EPOCH']-1) * \
                max(0, min(1, round(EPSILON(epoch))))
#        games = games[:min(n_games, len(games))]
#        
#        n_rounds = round(n_games/PARAMS['N_PAR_GAMES'])
#        for i in range(int(n_rounds)):
#            play_cgames(pool,
#                        games, 
#                        player, 
#                        EPSILON(epoch))
        
        for i in range(n_games):
            play_cgame(game, player, EPSILON(epoch))
    
        # Train
        player.train(PARAMS['MEM_BATCH'],
                     PARAMS['TRAIN_BATCH'], 
                     PARAMS['L_RATE'],
                     PARAMS['REG_RATE'],
                     PARAMS['MOM_RATE'])
        
        # If right epoch, play deterministic game
        if (epoch+1) % PARAMS['DET_VERBOSE'] == 0:
            
            turns, won, this_hist = play_cgame(game, player, -1, PARAMS['RAND_DETS'])
            
            print('Epoch %i (%i mins since (re-)start).' \
                  %(epoch+1,
                    (time.time() - start_time) / 60))
            
            if this_hist == last_hist:
                print('    Hist:', this_hist)
                print('           PLAYED THE SAME GAME!!!')
            last_hist = this_hist
            
            det_turnlist.append(N_PIECES-1 - turns)
            det_winlist.append(won)
            
        # Every SAVE_EVERY epochs, plot and save net
        if (epoch+1) % PARAMS['SAVE_EVERY'] == 0:
            
            save_progress(turnlist, winlist, 
                          det_turnlist, det_winlist,
                          PARAMS,
                          time_base + time.time() - start_time,
                          player,
                          DIRNAME+'/checkpoints', 
                          name_end = '_%i' %(epoch+1))
            plt.close()
            
    
    end_time = time.time()
    print('\nDone running RL! Saving...')
    
    # All done, save data and plot
    save_progress(turnlist, winlist, 
                  det_turnlist, det_winlist,
                  PARAMS,
                  time_base + end_time - start_time,
                  player,
                  DIRNAME)
    plt.close()
    
    # Delete checkpoints
    #shutil.rmtree(DIRNAME+'/checkpoints')
        
    
    
    
    
    
    
    
    
    
        
    
    

        
    