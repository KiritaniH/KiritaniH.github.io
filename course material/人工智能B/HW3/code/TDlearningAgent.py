    # TDlearningAgent.py
# -*- coding: utf-8 -*-
"""
TD(0) Policy Evaluation Agent for the provided Gridworld/MDP framework.

This agent performs **Temporal-Difference learning (TD(0))** to estimate a
state-value function V(s) for a **fixed policy π** while interacting with the
environment via sampling (episodes). It **does not** perform control (i.e.,
it does not try to improve the policy). Instead, it follows a given policy
and updates V(s) online from experienced transitions.

Integration expectations (based on the given framework):
- There exists an abstract base class `Agent` with at least `getAction(state)`.
- The episode runner will call:
    state = env.reset()
    while True:
        action = agent.getAction(state)
        if action is None: break
        nextState, reward = env.doAction(action)
        agent.observeTransition(state, action, nextState, reward)
        state = nextState
- For visualization, display code may call:
    - agent.getValue(state)
    - agent.getPolicy(state)
    - agent.getQValue(state, action)   # optional but useful for Q fan plots
- MDP conventions from the framework:
    - mdp.getStates()
    - mdp.getPossibleActions(state) -> tuple of str actions, e.g. ('north','south',...)
    - mdp.getTransitionStatesAndProbs(state, action) -> list of (nextState, prob)
    - mdp.getReward(state, action, nextState) -> float
    - mdp.isTerminal(state) -> bool
  NOTE: "Exit" squares are NOT terminal; they have the single action 'exit' that
  deterministically leads to the true terminal state.

Key design choices in this file:
- We require a **policy function** `policy_fn(state) -> action(str or None)`.
  The agent will follow this policy exactly (no exploration). This makes the
  algorithm a pure **TD(0) prediction** method.
- We maintain a value table `self.values` as a `util.Counter`, which returns 0
  by default for unseen states. This avoids KeyErrors and simplifies updates.
- We also implement `getQValue(s,a)` using the model (MDP dynamics) and the
  current V(s) for visualization convenience:
      Q(s,a) = sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]

Usage example (in main.py or wherever you construct the agent):
----------------------------------------------------------------
from TDlearningAgent import TDLearningAgent

def my_policy(state):
    # Example fixed policy: greedy to the East if possible, else North,
    # otherwise pick the first available action. Adapt to your needs.
    actions = mdp.getPossibleActions(state)
    if not actions:
        return None
    if 'east' in actions:
        return 'east'
    if 'north' in actions:
        return 'north'
    return actions[0]

agent = TDLearningAgent(
    mdp=mdp,
    discount=0.9,
    alpha=0.5
)

# Then run episodes with your existing loop / runEpisode(...) utility.

"""

from abstractAgent import Agent
import util
import random

def random_policy(mdp):
    return lambda s: None if mdp.isTerminal(s) else random.choice(mdp.getPossibleActions(s))
def greedy_policy(state):
    actions = mdp.getPossibleActions(state)
    if not actions:
        return None
    best_a = None
    best_r = float('-inf')
    for a in actions:
        for next_state, prob in mdp.getTransitionStatesAndProbs(state, a):
            reward = mdp.getReward(state, a, next_state)
            if reward > best_r:
                best_r = reward
                best_a = a
    return best_a

class TDLearningAgent(Agent):
    """
    TD(0) policy evaluation agent.

    This agent estimates the state-value function V(s) for a GIVEN policy π.
    It does NOT improve the policy. The environment interaction provides
    samples (s, a, r, s'), and we update V(s) online via:

        V(s) ← V(s) + α * [ r + γ * V(s') - V(s) ]

    where:
        α  (alpha)   : learning rate (0 < α ≤ 1)
        γ  (discount): discount factor (0 ≤ γ < 1)

    The action returned by getAction(s) is π(s) provided by `policy_fn`.
    """

    def __init__(self, mdp, discount=0.9, alpha=0.5):
        """
        Args:
            mdp:        An object implementing the MDP interface used in this project.
            discount:   Gamma in [0,1). Controls how much future returns are valued.
            alpha:      Learning rate for TD updates.
        """
        self.mdp = mdp
        self.discount = float(discount)
        self.alpha = float(alpha)
        #self.policy_fn = greedy_policy  # Default to a random policy; can be    overridden.
        def greedy_policy(state):
            actions = mdp.getPossibleActions(state)
            if not actions:
                return None
            best_a = None
            best_r = float('-inf')
            for a in actions:
                for next_state, prob in mdp.getTransitionStatesAndProbs(state, a):
                    reward = mdp.getReward(state, a, next_state)
                    if reward > best_r:
                        best_r = reward
                        best_a = a
            return best_a
        self.policy_fn = greedy_policy


        # V(s) table. util.Counter returns 0 for unseen keys by default.
        self.values = util.Counter()

        # (Optional) bookkeeping for episodes; your runner may call these.
        self.episode_returns = []   # list of cumulative (discounted) returns per episode (for analysis)
        self._G = 0.0               # running return within an episode
        self._t = 0                 # timestep within an episode

        

    # ===========================
    # Interfaces for visualization
    # ===========================

    def getValue(self, state):
        """
        Return the current estimate V(s) for a given state.
        Used by value visualization (heatmap/labels).
        """
        return self.values[state]
    
    def getPolicy(self, state):
        return self.policy_fn(state)

    def getQValue(self, state, action):
        """
        Return a one-step lookahead Q(s,a) under the current V(s),
        computed using the known model (MDP dynamics). This is only used
        for visualization (e.g., Q-fan plots). TD(0) learning itself does
        not require this model-based computation.

        Q(s,a) = sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]
        """
        # If no action is provided or state is terminal, Q is 0 by definition here.
        if self.mdp.isTerminal(state):
            return 0.0

        qvalue = 0.0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            qvalue += prob * (reward + self.discount * self.getValue(next_state))
        return qvalue

    # ===========================
    # Episode control hooks (optional)
    # ===========================

    def startEpisode(self):
        """
        Optional: Reset per-episode trackers (useful for logging learning curves).
        Callers (episode runner) may invoke this at the beginning of each episode.
        """
        self._G = 0.0
        self._t = 0

    def stopEpisode(self):
        """
        Optional: Push the accumulated return to a list for analysis.
        Callers may invoke this at the end of each episode.
        """
        self.episode_returns.append(self._G)

    # ===========================
    # Acting & Learning
    # ===========================
    
    def getAction(self, state):
        return self.policy_fn(state)

    def observeTransition(self, state, action, nextState, reward):
        """
        TODO: implement the TD(0) update rule here.
        V(s) ← V(s) + α * [r + γ * V(s') - V(s)]
        """
        # ====== YOUR CODE HERE ======
        # hint: use self.getValue(nextState) to get V(s')
        # and self.values[state] for current V(s)
        # don't forget to use self.alpha and self.discount

        current_value = self.values[state]
        
        next_value = self.getValue(nextState)
        
        td_target = reward + self.discount * next_value
        
        td_error = td_target - current_value
        
        self.values[state] = current_value + self.alpha * td_error

        # =============================
        # (Optional) bookkeeping for returns, using discounted sum of rewards
        # G_t = r_t + gamma * G_{t+1}; here we accumulate forward for logging.
        # This does not affect learning; it's only for analysis.
        self._G += (self.discount ** self._t) * reward
        self._t += 1

    