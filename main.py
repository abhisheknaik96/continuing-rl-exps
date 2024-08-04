import argparse
import os
import time
import traceback
import numpy as np

from utils.sweeper import Sweeper
from utils.helpers import validate_output_folder
from experiments import run_experiment_one_config

parser = argparse.ArgumentParser(description="Run an experiment based on parameters specified in a configuration file")
parser.add_argument('--config-file',  # required=True,
                    default='config_files/randomwalk5/test.json',
                    help='location of the config file for the experiment (e.g., config_files/test_config.json)')
parser.add_argument('--cfg-start', default=0)
parser.add_argument('--cfg-end', default=-1)
parser.add_argument('--output-path', default='results/test_exp/')
args = parser.parse_args()
print(args.config_file, args.output_path)
path = validate_output_folder(args.output_path)

sweeper = Sweeper(args.config_file)
cfg_start_idx = int(args.cfg_start)
cfg_end_idx = int(args.cfg_end) if args.cfg_end != -1 else sweeper.total_combinations

print(f'\n\nRunning configurations {cfg_start_idx} to {cfg_end_idx}...\n\n')

start_time = time.time()

for i in range(cfg_start_idx, cfg_end_idx):
    config = sweeper.get_one_config(i)
    config['exp_id'] = i
    config['output_folder'] = path
    # print(f'Starting at: {time.localtime(start_time)}')
    print(config)

    try:
        log = run_experiment_one_config(config)
        log['params'] = config
    except Exception as e:
        print('\n***\n')
        print(traceback.format_exc())
        print('***\nException occurred with this parameter configuration, moving on now\n***\n')
    else:
        filename = f"{config['exp_name']}_{config['exp_id']}"
        print(f'Saving experiment log in: {filename}.npy\n**********\n')
        np.save(f'{path}{filename}', log)
    finally:
        print("Time elapsed: {:.2} minutes\n\n".format((time.time() - start_time) / 60))
        os.system('sleep 0.5')

end_time = time.time()
print("Total time elapsed: {:.2} minutes".format((end_time - start_time) / 60))
