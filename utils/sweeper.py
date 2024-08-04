# Based on the sweeper.py file in
# https://github.com/muhammadzaheer/classic-control/blob/0f075ee2951de01d063bc1d069b28bf25167af20/sweeper.py
import copy
import json


class Sweeper:
    """Class to help perform hyperparameter sweeps.

    This class takes a json file as input
    and creates one dictionary per unique configuration of hyperparameters.
    """
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config_dict = json.load(f)
        self.total_combinations = 1

        sweep_params = self.config_dict['sweep_parameters']
        # calculating total_combinations
        tc = 1
        for params, values in sweep_params.items():
            tc = tc * len(values)
        self.total_combinations = tc

    def get_one_config(self, idx):
        """replaces the range of values by a single value based on the index idx"""
        cfg = copy.deepcopy(self.config_dict)
        sweep_params = cfg.pop('sweep_parameters')
        cumulative = 1
        for param, values in sweep_params.items():
            cfg[param] = values[int(idx/cumulative) % len(values)]
            cumulative *= len(values)
        return cfg


if __name__ == '__main__':
    sweeper = Sweeper("../config_files/test_config_prediction.json")
    print(f'Total combinations: {sweeper.total_combinations}')
    for i in range(sweeper.total_combinations):
        print(sweeper.get_one_config(i))
