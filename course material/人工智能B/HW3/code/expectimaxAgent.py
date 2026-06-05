from abstractAgent import Agent
import random

class ExpectimaxAgent(Agent):
    """An expectimax agent with search trees iterate to a fixed depth.

    Hints:
    1) Use functions of self.mdp to get necessary information;
    2) A search with depth limit `d` considers `d` maximizer nodes and `d` chance nodes.
    3) Test your code with commands like `python main.py -a expectimax --depth 3 -k 10`.
    """

    def __init__(self, mdp, depth):
        self.mdp = mdp
        self.depth = depth

    def maximize(self, state, d):
        if self.mdp.isTerminal(state):
            return 0, None

        available_actions = self.mdp.getPossibleActions(state)

        if available_actions[0] == "exit":
            terminal_state = self.mdp.getTransitionStatesAndProbs(state, "exit")[0][0]
            assert self.mdp.getTransitionStatesAndProbs(state, "exit")[0][1] == 1
            reward = self.mdp.getReward(state, "exit", terminal_state)
            return reward, "exit"

        """ YOUR CODE HERE """
        action_utility = [] # [0 for _ in range(len(available_actions))]
        for action in available_actions:
            utility = self.chance(state, action, d)
            action_utility.append(utility)

        maximum = max(utility for utility in action_utility)
        max_action = random.choice([
            available_actions[k] for k in range(len(action_utility)) if action_utility[k] == maximum
        ])
        return maximum, max_action

    def chance(self, state, action, d):
        # To keep it simple, we use zero as the estimated values when the depth limit is reached.
        if d >= self.depth:
            return 0

        """ YOUR CODE HERE """
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        utility = 0
        for transition in transitions:
            next_state = transition[0]
            probability = transition[1]
            u, _ = self.maximize(next_state, d + 1)
            utility += u * probability

        return utility

    def getAction(self, state):
        _, act = self.maximize(state, 0)
        return act

    def getValue(self, state):
        value, _ = self.maximize(state, 0)
        return value

    def getPolicy(self, state):
        _, act = self.maximize(state, 0)
        return act