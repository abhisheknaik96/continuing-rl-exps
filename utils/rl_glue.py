
class RLGlue:
    """Glues together an experiment, agent, and environment."""

    def __init__(self, env_class, agent_class, **parameters):
        """Initializes the rl-glue interface.
        Args:
            env_class: name of the environment
            agent_class: name of the agent
            **parameters: parameters for the environment and the agent
        """
        self.environment = env_class(*parameters)
        self.agent = agent_class(*parameters)
        self.last_action = None
        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self):
        """Starts RLGlue experiment

        Returns:
            tuple: (state, action)
        """
        last_state = self.environment.start()
        self.last_action = self.agent.start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation):
        """Starts the agent.

        Args:
            observation: The first observation from the environment
        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent

        Args:
            reward (float): the last reward the agent received for taking the
                last action.
            observation : the state observation the agent receives from the
                environment.
        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_step(reward, observation)


    def rl_env_start(self):
        """Starts RL-Glue environment.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent

        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def rl_step(self):
        """Step taken by RLGlue, takes environment step and either step or
            end by agent.

        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """

        (reward, last_state, term) = self.environment.env_step(self.last_action)

        self.total_reward += reward

        # if term:
        #     self.num_episodes += 1
        #     self.agent.agent_end(reward)
        #     roat = (reward, last_state, None, term)
        # else:
        #     self.num_steps += 1
        #     self.last_action = self.agent.agent_step(reward, last_state)
        #     roat = (reward, last_state, self.last_action, term)
        self.num_steps += 1
        self.last_action = self.agent.agent_step(reward, last_state, term)
        roat = (reward, last_state, self.last_action, term)

        return roat
