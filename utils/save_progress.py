## ~~~~~~~~~~~~~
#  Save progress
## ~~~~~~~~~~~~~
    
import json
import matplotlib.pyplot as plt


def save_progress(det_res, 
                  params, e_str, l_str,
                  time_elapsed, player,
                  save_dir, name_end = ''):
    
    # Save params
    with open(save_dir + '/results%s.json' %name_end, 'w') as file:
        json.dump({'det_res': det_res,
                   'time': time_elapsed}, 
                  file)
    
    # Save network
    player.network.save(save_dir + '/network%s.json' %name_end)
    
    # Plot and save progress
    fig, ax = plt.subplots()

    for i in range(len(det_res)):
        ax.plot(i * params['DET_VERBOSE'], det_res[i], 
                'g.' if det_res[i] == 1 else 'r.',
                markersize = 1)

    ax.set_xlabel('Epoch, i')
    ax.set_ylabel('Pieces Left')
    ax.set_title('RL Progress\n'+ \
                 ('Full-Sized Board' if params['BOARD_SIZE'] in (-1, 4) else \
                  'Board Size: %i/5' % (params['BOARD_SIZE']+1)))

    text = 'ε(i): %s \n' \
           'games(i): 1 %s\n\n' \
           'm batch: %i/%i \n' \
           't batch: %i \n\n' \
           'learn(i): %s \n' \
           'reg: %0.4f \n' \
           'mom: %0.2f \n\n' \
           'time: %0.1f mins' \
            % (e_str,
               '' if params['GAMES_PER_EPOCH']==1 else \
                    '+ %i max(0, min(1, ε(i)))' %(params['GAMES_PER_EPOCH']-1),
               params['MEM_BATCH'], params['MAX_MEM_LEN'],
               round(params['TRAIN_BATCH'] * params['MEM_BATCH']),
               l_str,
               params['REG_RATE'],
               params['MOM_RATE'], 
               time_elapsed/60)

    ax.text(0.98, 0.98, text,
            transform = ax.transAxes, fontsize = 9,
            verticalalignment = 'top',
            horizontalalignment = 'right')
    
    plt.show(block = False)
    plt.savefig(save_dir + '/progress%s.pdf' %name_end)