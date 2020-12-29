## ~~~~~~~~~~~~~~~~~~
#  Main file
## ~~~~~~~~~~~~~~~~~~


'''

Reads parameters and progress data from json files, 
does setup (initialise classes, etc.)
and finally runs the RL routine.

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

EPSILON = lambda i: sum([PARAMS['EPSILON'][j] * i**j \
                         for j in range(len(PARAMS['EPSILON']))])
L_RATE = lambda i: sum([PARAMS['L_RATE'][j] * i**j \
                         for j in range(len(PARAMS['L_RATE']))])

game = game.Game(PARAMS['BOARD_SIZE'])
PLAYER_CLASS = getattr(players, PARAMS['PLAYER_TYPE'])

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
print('Training via RL')
print('~~~~~~~~~~~~~~~')
print('\033[0m')

l_str = ''.join([(str(PARAMS['L_RATE'][i]) if i == 0 \
                  else ' + '+str(PARAMS['L_RATE'][i]) if PARAMS['L_RATE'][i]>=0 else \
                  ' - '+str(-PARAMS['L_RATE'][i])) + \
                 ('' if i==0 else ' i' if i==1 else ' i^'+str(i)) \
                 for i in range(len(PARAMS['L_RATE']))])

e_str = ''.join([(str(PARAMS['EPSILON'][i]) if i == 0 \
                  else ' + '+str(PARAMS['EPSILON'][i]) if PARAMS['EPSILON'][i]>=0 else \
                  ' - '+str(-PARAMS['EPSILON'][i])) + \
                 ('' if i==0 else ' i' if i==1 else ' i^'+str(i)) \
                 for i in range(len(PARAMS['EPSILON']))])

print('Board size: %i/5' %(PARAMS['BOARD_SIZE']+1), '\n\n',
      
      'Parameters:',
      '\n - network inner shape :', PARAMS['INNER_SHAPE'], '\n',
      
      '\n - discount rate       :', PARAMS['DISC_RATE'],
      '\n - max memory len      :', PARAMS['MAX_MEM_LEN'], '\n',
      
      '\n - memory batch size   :', PARAMS['MEM_BATCH'],
      '\n - training batch size :', PARAMS['TRAIN_BATCH'], '\n',
      
      '\n - learning rate (i)   :', l_str,
      '\n - regularisation rate :', PARAMS['REG_RATE'],
      '\n - momentum rate       :', PARAMS['MOM_RATE'], '\n',
      
      '\n - epochs              :', PARAMS['NUM_EPOCHS'], 
      '\n - epsilon(i)          :', e_str,
      '\n - games/epoch (i)     : 1 + %i max(0, min(1, ε(i)))' % (PARAMS['GAMES_PER_EPOCH']-1), 
      '\n')


## Reinforcement learning
## ~~~~~~~~~~~~~~~~~~~~~~

last_hist = ''
start_time = time.time()
for epoch in range(CURRENT_EPOCH, PARAMS['NUM_EPOCHS']):

    # Play games
    n_games = 1 + (PARAMS['GAMES_PER_EPOCH']-1) * \
                    round(max(0, min(1, EPSILON(epoch))))

    for i in range(n_games):
        player.play(game, EPSILON(epoch))

    # Train
    player.train(PARAMS['MEM_BATCH'],
                 PARAMS['TRAIN_BATCH'], 
                 L_RATE(epoch),
                 PARAMS['REG_RATE'],
                 PARAMS['MOM_RATE'])

    # If right epoch, play deterministic game
    if (epoch+1) % PARAMS['DET_VERBOSE'] == 0:

        res, this_hist = player.play(game, -1, PARAMS['RAND_DETS'])

        print('Epoch %i (%i mins).' \
              %(epoch+1,
                (time_base + time.time() - start_time) / 60))

        if this_hist == last_hist:
            print('  PLAYED THE SAME GAME AS LAST TRY!!!')
            print('  Hist:', this_hist)
        last_hist = this_hist

        det_res.append(res)

    # Every SAVE_EVERY epochs, plot and save net
    if (epoch+1) % PARAMS['SAVE_EVERY'] == 0:

        save_progress(det_res, 
                      PARAMS, EPSILON, L_RATE,
                      time_base + time.time() - start_time,
                      player,
                      DIRNAME, 
                      temp = True)
        plt.close()


end_time = time.time()
print('\nDone running RL! Saving...')

# All done, save data and plot
save_progress(det_res,
              PARAMS, EPSILON, L_RATE,
              time_base + end_time - start_time,
              player,
              DIRNAME)
#input('Press Enter to close plot and finish.')
plt.close()

# Remove temp files
os.remove(DIRNAME + '/progress_temp.pdf')
os.remove(DIRNAME + '/network_temp.json')
os.remove(DIRNAME + '/results_temp.json')
    
    
    
    
    
    
    
    
    
        
    
    

        
    