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

from testRO import *    


## Do RL
## ~~~~~

if __name__ == "__main__":
    
    import time, re, shutil
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n     Test - RO     \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    
    ## Load params
    ## ~~~~~~~~~~~
    
    PLAYER_CLASS = QPlayer
    DIRNAME = 'testMRO_saves/RO/test3'
        # ^ DO NOT ADD '/' to end!
    
    # Get params
    with open(DIRNAME + '/params.json', 'r') as file:
        PARAMS = json.load(file)
    
    # Rename folder if set to
    if PARAMS['RENAME']:
        
        NEW_DIRNAME = DIRNAME.split('/')
        NEW_DIRNAME[-1] = 'e%ik g%i mb%i tb%i lr%s rr%s mr%s %srand' \
            %(PARAMS['NUM_EPOCHS']/1e3,
              PARAMS['GAMES_PER_EPOCH'],
              PARAMS['MEM_BATCH'],
              PARAMS['TRAIN_BATCH'] * PARAMS['MEM_BATCH'],
              str(PARAMS['L_RATE']).replace('.', '-'),
              str(PARAMS['REG_RATE']).replace('.', '-'),
              str(PARAMS['MOM_RATE']).replace('.', '-'),
              '' if PARAMS['RAND_DETS'] else 'no')
        NEW_DIRNAME = '/'.join(NEW_DIRNAME)
        
        os.rename(DIRNAME, NEW_DIRNAME)
        DIRNAME = NEW_DIRNAME
        
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

        # Get progress data
        with open(DIRNAME + '/checkpoints/results_temp.json', 'r') as file:
            data = json.load(file)
        
        turnlist = data['turnlist']
        winlist = data['winlist']

        det_turnlist = data['det_turnlist']
        det_winlist = data['det_winlist']
        
        CURRENT_EPOCH = len(det_turnlist)*PARAMS['DET_VERBOSE']
        
        time_base = data['time']
        
        # Set up player
        player = PLAYER_CLASS(PARAMS['MAX_MEM_LEN'],
                          PARAMS['DISC_RATE'],
                          DIRNAME + '/checkpoints/network_temp.json')
    
    # Final stuff
    EPSILON = lambda i: sum([PARAMS['EPSILON'][j] * i**j \
                             for j in range(len(PARAMS['EPSILON']))])
    
    game  =  Game()
    
    
    # Reinforcement learning
    # ~~~~~~~~~~~~~~~~~~~~~~
    
    last_hist = ''
    start_time = time.time()
    for epoch in range(CURRENT_EPOCH, PARAMS['NUM_EPOCHS']):
        
        # Play games
        n_games = 1 + \
             (PARAMS['GAMES_PER_EPOCH']-1) * \
                max(0, min(1, round(EPSILON(epoch))))
                
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
                    (time_base + time.time() - start_time) / 60))
            
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
                          name_end = '_temp')
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
        
    
    
    
    
    
    
    
    
    
        
    
    

        
    