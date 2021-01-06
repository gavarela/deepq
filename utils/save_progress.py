## ~~~~~~~~~~~~~
#  Save progress
## ~~~~~~~~~~~~~
    
import json
import matplotlib.pyplot as plt


def save_progress(det_res, network, 
                  params,
                  time_elapsed,
                  save_dir,
                  temp = False,
                  epsilon = None, lrate = None):
    
    name_end = '_temp' if temp else ''
    
    # Save results
    with open(save_dir + '/results%s.json' %name_end, 'w') as file:
        json.dump({'det_res': det_res,
                   'time': time_elapsed}, 
                  file)
    
    # Save network
    network.save(save_dir + '/network%s.json' %name_end)
    
    # Plot and save progress
    if temp:
        
        fig, ax = plt.subplots()
        for i in range(len(det_res)):
            ax.plot(i * params['DET_VERBOSE'], det_res[i], 
                    'g.' if det_res[i] == 1 else 'r.',
                    markersize = 1)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pieces Left')
        ax.set_title('RL Progress\n'+ \
                     ('Full-Sized Board' if params['BOARD_SIZE'] == 5 else \
                      'Board Size: %i/5' % params['BOARD_SIZE']))
        
        text = 'ε(i): %s \n' \
               'learn(i): %s \n\n' \
               'train batches: %i/%i/%i x%i \n' \
               'games/i: %i \n' \
               'reg, mom: %0.4f, %0.2f \n\n' \
               'time: %0.1f mins' \
                % (epsilon, lrate,
                   params['TRAIN_BATCH'], params['MEM_BATCH'], 
                   params['MAX_MEM_LEN'], params['TRAINS_PER_EPOCH'],
                   params['GAMES_PER_EPOCH'], 
                   params['REG_RATE'], params['MOM_RATE'], 
                   time_elapsed/60)
                
        ax.text(0.98, 0.98, text,
                transform = ax.transAxes, fontsize = 9,
                verticalalignment = 'top',
                horizontalalignment = 'right')
    
    else:
        
        # Prep grid of plots
        fig, axs = plt.subplots(3, 6, figsize = [10, 4.8])
        gs = axs[0, 0].get_gridspec()
        for row in axs:
            for ax in row:
                ax.remove()

        bigax = fig.add_subplot(gs[:, :-2])
        smallax1 = fig.add_subplot(gs[0, -2:])
        smallax2 = fig.add_subplot(gs[1, -2:])
        smallax3 = fig.add_subplot(gs[2, -2:])

        # Plot progress (main axis)
        for i in range(len(det_res)):
            bigax.plot(i * params['DET_VERBOSE'], det_res[i], 
                       'g.' if det_res[i] == 1 else 'r.',
                       markersize = 1)

        bigax.set_xlabel('Epoch')
        bigax.set_ylabel('Pieces Left')
        bigax.set_title('RL Progress\n'+ \
                        ('Full-Sized Board' if params['BOARD_SIZE'] == 5 else \
                         'Board Size: %i/5' % params['BOARD_SIZE']))

        text = 'Net. inner shape: (%s) \n\n' \
               'Batches: %i/%i/%i \n' \
               'Trains/i: %i \n\n' \
               'Reg. rate: %0.4f \n' \
               'Mom. rate: %0.2f \n\n' \
               'Time: %0.1f mins' \
                % (', '.join([str(i) for i in params['INNER_SHAPE']]),
                   params['TRAIN_BATCH'], params['MEM_BATCH'], params['MAX_MEM_LEN'],
                   params['TRAINS_PER_EPOCH'],
                   params['REG_RATE'],
                   params['MOM_RATE'], 
                   time_elapsed/60)

        fig.text(-1.5, 3.2, text,
                 transform = ax.transAxes, fontsize = 9,
                 verticalalignment = 'top',
                 horizontalalignment = 'right')

        # Smallplots in a row
        epochs = [i * params['DET_VERBOSE'] for i in range(len(det_res))]

        smallax1.plot(epochs, [max(0, min(1, epsilon[i])) for i in epochs],
                      markersize = 1)
        smallax1.set_ylabel('Rand. prob.')
        smallax1.set_ylim([-0.1, 1.1])

        smallax2.plot(epochs, [1 + round((params['GAMES_PER_EPOCH']-1) * epsilon[i]) for i in epochs],
                      markersize = 1)
        smallax2.set_ylabel('Num. games')
        smallax2.set_ylim([0.9, params['GAMES_PER_EPOCH']+0.1])

        smallax3.plot(epochs, [lrate[i] for i in epochs],
                      markersize = 1)
        smallax3.set_ylabel('Learn. rate')
        smallax3.set_xlabel('Epoch')

        # Finish up
        fig.tight_layout()
        
    plt.show(block = False)
    plt.savefig(save_dir + '/progress%s.pdf' %name_end)
    plt.savefig(save_dir + '/progress%s.jpg' %name_end)