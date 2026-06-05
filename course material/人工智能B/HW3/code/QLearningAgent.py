# QLearningAgent.py
# -*- coding: utf-8 -*-
"""
Off-policy Q-Learning agent for the provided Gridworld/MDP framework.

This agent learns an action-value function Q(s,a) from sampled transitions
without requiring the model (though we still use mdp.getPossibleActions to
know legal actions). Behavior is epsilon-greedy w.r.t. current Q.

Update rule (per transition):
    Q(s,a) ← Q(s,a) + α * [ r + γ * max_{a'} Q(s',a') - Q(s,a) ]

Integration expectations (same as TDLearningAgent):
- The episode runner calls:
    action = agent.getAction(state)
    nextState, reward = env.doAction(action)
    agent.observeTransition(state, action, nextState, reward)
- For visualization the display may call:
    getQValue(s,a), getValue(s), getPolicy(s)
- Actions are strings like 'north','south','east','west','exit'.
"""

from abstractAgent import Agent
import util, random


class QLearningAgent(Agent):
    """
    Off-policy Q-learning with epsilon-greedy behavior.
    Maintains a table Q[(s,a)] using util.Counter (default 0 for unseen keys).
    """

    def __init__(self, mdp, discount=0.9, alpha=0.5, epsilon=0.1):
        """
        Args:
            mdp:        MDP object used to query legal actions and terminal states.
            discount:   Gamma in [0,1).
            alpha:      Learning rate in (0,1].
            epsilon:    Exploration rate for epsilon-greedy behavior.
        """
        self.mdp = mdp
        self.discount = float(discount)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

        # Q-value table with default 0 for unseen (state, action) pairs.
        self.qvalues = util.Counter()

        # Optional: stats for plotting/analysis
        self.episode_returns = []
        self._G = 0.0
        self._t = 0

    # --------- Value/Q/Policy (for display) ---------

    def getQValue(self, state, action):
        """Return learned Q(s,a); defaults to 0 for unseen pairs."""
        return self.qvalues[(state, action)]

    def getValue(self, state):
        """
        V(s) = max_a Q(s,a); returns 0.0 when no legal actions (terminal).
        Useful for value heatmap.
        """
        if self.mdp.isTerminal(state):
            return 0.0
        
        all_actions = self.mdp.getPossibleActions(state)
        return max(self.getQValue(state, action) for action in all_actions)

    def getPolicy(self, state):
        """
        TODO: Implement Greedy policy Q for visualization.
        """
        # ====== YOUR CODE HERE ======
        if self.mdp.isTerminal(state):
            return None
        
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        
        best_action = None
        best_q_value = float('-inf')
        
        for action in actions:
            q_value = self.getQValue(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action
        
    # --------- Acting (epsilon-greedy) ---------

    def computeAction(self, state):
        """
        TODO: Implement epsilon-greedy action selection.
        With probability ε: choose random action.
        With probability (1-ε): choose argmax_a Q(s,a).
        """
        # ====== YOUR CODE HERE ======
        if self.mdp.isTerminal(state):
            return None
        
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return self.getPolicy(state)
        
        return self.getPolicy(state)

    def getAction(self, state):
        return self.computeAction(state)

    # --------- Learning ---------

    def observeTransition(self, state, action, nextState, reward):
        """
        TODO: Implement the Q-learning update.
        Formula:
            Q(s,a) ← Q(s,a) + α * [r + γ * max_{a'} Q(s',a') - Q(s,a)]
        """
        # ====== YOUR CODE HERE ======
        current_q = self.getQValue(state, action)
        
        if self.mdp.isTerminal(nextState):
            max_next_q = 0
        else:
            next_actions = self.mdp.getPossibleActions(nextState)
            max_next_q = max(self.getQValue(nextState, a) for a in next_actions)
        
        target = reward + self.discount * max_next_q
        
        td_error = target - current_q
        
        self.qvalues[(state, action)] = current_q + self.alpha * td_error

        # Optional stats
        self._G += (self.discount ** self._t) * reward
        self._t += 1

    # --------- Optional episode hooks ---------

    def startEpisode(self):
        self._G = 0.0
        self._t = 0

    def stopEpisode(self):
        self.episode_returns.append(self._G)