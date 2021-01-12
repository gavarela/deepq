## ~~~~~~~~~~~~~~~~~~
#  Main file
## ~~~~~~~~~~~~~~~~~~


'''

Reads parameters and progress data from json files, 
does setup (initialise classes, etc.)
and runs the RL routine.

'''

import sys, time, re, json, os

import game, players
from utils import *


## Print colourful title
## ~~~~~~~~~~~~~~~~~~~~~

print('\033[33m')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('     Remain One (Peg Solitaire)   ')
print('      Reinforcement Q-Learning    ')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('\033[0m')


## Load parameters and do setup
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DIRNAME = sys.argv[1].rstrip('/')

# Get params
with open(DIRNAME + '/params.json', 'r') as file:
    PARAMS = json.load(file)

EPSILON = interp(list(range(PARAMS['NUM_EPOCHS'])), PARAMS['EPSILON'], PARAMS['NUM_EPOCHS'])
L_RATE  = interp(list(range(PARAMS['NUM_EPOCHS'])), PARAMS['L_RATE'],  PARAMS['NUM_EPOCHS'])

if not isinstance(PARAMS['LTERM_PROPS'], list) and isinstance(PARAMS['LTERM_TIMES'], list):
    PARAMS['LTERM_PROPS'] = [PARAMS['LTERM_PROPS'] for i in range(len(PARAMS['LTERM_TIMES']))]
PARAMS['LTERM_TIMES'] += [2]

game = game.Game(PARAMS['BOARD_SIZE']-1)
PLAYER_CLASS = getattr(players, PARAMS['PLAYER_TYPE'])

# Setup player and network (from checkpoint, if applicable)
filenames = os.listdir(DIRNAME)
resume = 'progress_temp.pdf' in filenames

if not resume:

    SHAPE = [game._TOT_PIECES] + \
            PARAMS['INNER_SHAPE'] + \
            [len(game._MOVES)]
    player = PLAYER_CLASS(PARAMS['MAX_MEM_LEN'],
                          PARAMS['DISC_RATE'],
                          SHAPE)

    # For training
    det_res = []
    time_base = 0

    CURRENT_EPOCH = 0

else:

    # Get progress data
    with open(DIRNAME + '/results_temp.json', 'r') as file:
        data = json.load(file)

    det_res = data['det_res']
    time_base = data['time']
    
    CURRENT_EPOCH = len(det_res) * PARAMS['DET_VERBOSE'] + 1

    # Set up player
    player = PLAYER_CLASS(PARAMS['MAX_MEM_LEN'],
                          PARAMS['DISC_RATE'],
                          DIRNAME + '/network_temp.json')


## Print params
## ~~~~~~~~~~~~

print('\033[32m')
print('Board size: %i/5' %PARAMS['BOARD_SIZE'], '\n\n',
      
      'Parameters:',
      '\n - network inner shape :', PARAMS['INNER_SHAPE'], '\n',
      
      '\n - discount rate       :', PARAMS['DISC_RATE'],
      '\n - max memory len      :', PARAMS['MAX_MEM_LEN'], 
      '\n - lterm memory props  :', str(PARAMS['LTERM_PROPS']), 
      '\n - lterm memory times  :', str(PARAMS['LTERM_TIMES']), '\n',
      
      '\n - memory batch size   :', PARAMS['MEM_BATCH'],
      '\n - training batch size :', PARAMS['TRAIN_BATCH'], '\n',
      
      '\n - learning rate (i)   :', str(PARAMS['L_RATE']),
      '\n - regularisation rate :', PARAMS['REG_RATE'],
      '\n - momentum rate       :', PARAMS['MOM_RATE'], '\n',
      
      '\n - epochs              :', PARAMS['NUM_EPOCHS'], 
      '\n - epsilon(i)          :', str(PARAMS['EPSILON']), '\n')
print('\033[0m')

## Reinforcement learning
## ~~~~~~~~~~~~~~~~~~~~~~

last_hist = ''
start_time = time.time()
for epoch in range(CURRENT_EPOCH, PARAMS['NUM_EPOCHS']):

    # Play game
    player.play(game, EPSILON[epoch])

    # Train
    for i in range(PARAMS['TRAINS_PER_EPOCH']):
        player.train(PARAMS['MEM_BATCH'],
                     PARAMS['TRAIN_BATCH'], 
                     L_RATE[epoch],
                     PARAMS['REG_RATE'],
                     PARAMS['MOM_RATE'])
    
    # If right epoch, commit some prop of memory to long term mem
    if (epoch+1) == int(PARAMS['NUM_EPOCHS']*PARAMS['LTERM_TIMES'][0]):
        player.remember(PARAMS['LTERM_PROPS'][0])
        PARAMS['LTERM_TIMES'] = PARAMS['LTERM_TIMES'][1:]
        PARAMS['LTERM_PROPS'] = PARAMS['LTERM_PROPS'][1:]

    # If right epoch, play deterministic game
    if (epoch+1) % PARAMS['DET_VERBOSE'] == 0:

        res, this_hist = player.play(game, -1, PARAMS['RAND_DETS'])

        print('Epoch %i (%i mins).' \
              %(epoch+1,
                (time_base + time.time() - start_time) / 60))

        if this_hist == last_hist:
            print('  PLAYED SAME GAME:', this_hist)
        last_hist = this_hist

        det_res.append(res)

    # Every SAVE_EVERY epochs, plot and save net
    if (epoch+1) % PARAMS['SAVE_EVERY'] == 0:

        save_progress(det_res, player.network,
                      PARAMS,
                      time_base + time.time() - start_time,
                      DIRNAME, 
                      temp = True,
                      epsilon = EPSILON, 
                      lrate = L_RATE)
        plt.close()


end_time = time.time()
print('\nDone running RL! Saving...')

# All done, save data and plot
save_progress(det_res, player.network,
              PARAMS, 
              time_base + time.time() - start_time,
              DIRNAME, 
              epsilon = EPSILON, 
              lrate = L_RATE)
#input('Press Enter to close plot and finish.')
plt.close()

# Remove temp files
os.remove(DIRNAME + '/progress_temp.pdf')
os.remove(DIRNAME + '/progress_temp.jpg')
os.remove(DIRNAME + '/network_temp.json')
os.remove(DIRNAME + '/results_temp.json')
    
    
    
    
    
    
    
    
    
        
    
    

        
    