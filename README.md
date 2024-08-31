# continuing-rl-exps

Code for running reinforcement-learning (RL) experiments on continuing (non-episodic) problems.

This repository contains code for (1) different RL algorithms, (2) some environments, and (3) the agent-env loop to run experiments with different parameters and multiple runs.

### Code organization

- `agents/`: 
    - Prediction algorithms: 
        - Differential TD (Wan, Naik, Sutton, 2021)
        - Differential TD(lambda) (Naik, Sutton, 2022; Naik, 2024)
        - Discounted TD(lambda) (Sutton, 1988)
        - Discounted TD(lambda) with reward centering (Naik, Wan, Tomar, Sutton, 2024; Naik, 2024)
        - Average-cost TD(lambda) (Tsitsklis, Van Roy, 1999)
    - Control algorithms:
        - Discounted Q-learning (Watkins, Dayan, 1992)
        - Discounted Sarsa (Rummery, Niranjan, 1994)
        - Discounted Q-learning with reward centering (Naik, Wan, Tomar, Sutton, 2024; Naik, 2024)
        - Discounted Sarsa with reward centering
        - Differential Q-learning (Wan, Naik, Sutton, 2021)
- `environments/`:
    - Some multi-armed bandits
    - Acrobot
    - An n-state random walk (Naik, Sutton, 2022)
    - A couple other simple diagnostic environments
    - AccessControl, Catch, Puckworld, and other continuing environments can be run from [my fork](https://github.com/abhisheknaik96/csuite/) of Zhao et al.'s (2022) csuite.
- `config_files/`: JSON files containing all the parameters required to run a particular experiment
- `utils/`: various utilities and helper functions
- `experiments.py`: contains the agent-environment interaction loop
- `main.py`: used to start an experiment based on the parameters specified in `config_files`

An example experiment can be run using:
```
python main.py --config-file='config_files/accesscontrol/test.json' --output-path='results/test_exp/'
```
Some basic plotting code is in `plot_results_example.ipynb`.


### Types of function approximation supported

The prediction algorithms can be run with linear function approximation (using tile coding (see Sutton & Barto (2018): Section 9.5.4)) and tabular representations (via a one-hot encoding).

The control algorithms can be run with tabular, linear, and non-linear function approximation. 
The non-linear algorithms are essentially Mnih et al.'s (2015), and Naik, Wan, Tomar, Sutton's (2024) DQN with reward centering and the differential version of DQN.


### One algorithm implementation for different algorithms

There is a single algorithmic implementation which results in the different algorithms with different parameter choices.
For example, for the control algorithms, there is one implementation of a discounted algorithm with reward centering. Then:
- $\gamma \in [0,1), \eta=0$: Discounted Q-learning 
- $\gamma \in [0,1), \eta > 0$: Discounted Q-learning with reward centering
- $\gamma=1, \eta > 0$: Differential Q-learning

---

The code in this repository can run most—if not all—experiments in the following works:
- Naik, 2024: _Reinforcement Learning for Continuing Problems Using Average Reward,_ Ph.D. Dissertation, University of Alberta. [[Link](https://era.library.ualberta.ca/items/42307739-a774-4d6b-b1a3-de9fbc949575)]
- Naik, Wan, Tomar, Sutton, 2024: _Reward Centering,_ RLC. [[Link](https://arxiv.org/abs/2405.09999)]
- Naik, Sutton, 2022: _Multi-Step Average-Reward Prediction via Differential TD(lambda),_ RLDM. [[Link](https://drive.google.com/file/d/1Ygx2XI0mOFw4pfWZOm67uWsJQTOZEPSj/view?usp=sharing)]
- Wan*, Naik*, Sutton, 2021: _Learning and Planning in Average-Reward Markov Decision Processes,_ ICML. [[Link](https://arxiv.org/abs/2006.16318)]

Note: Instead of maintaining multiple public repositories on github for all the different projects in my PhD, I created this single repository that can probably run every experiment in my dissertation.
However, I have not re-run all those experiments with this unified codebase. 
If you are experiencing some unexpected results, feel free to reach out to me at `abhisheknaik22296@gmail.com` and I will be happy to work those out with you :)
