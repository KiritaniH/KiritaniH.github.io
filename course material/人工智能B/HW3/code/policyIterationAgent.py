import util
from abstractAgent import Agent

class PolicyIterationAgent(Agent):
    """An agent that takes a Markov decision process on initialization
    and runs policy iteration for a given number of iterations.

    Hint: Test your code with commands like `python3 main.py -a policy -i 100 -k 10`.
    """

    def __init__(self, mdp, discount = 0.9, epsilon=0.001, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of policy iteration
        self.iterations = iterations # The policy iteration will run AT MOST these steps
        self.values = util.Counter() # You need to keep the record of all state values here
        self.policy = dict()
        self.runPolicyIteration()

    def runPolicyIteration(self):
        states = self.mdp.getStates()
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                if actions:
                    self.policy[state] = actions[0] 
        
        for i in range(self.iterations):
            self.policyEvaluation()
            
            policy_changed = self.policyImprovement()
            
            if not policy_changed:
                print(f"Policy iteration converged after {i+1} iterations")
                break

    def policyEvaluation(self):
        for _ in range(self.iterations):
            max_change = 0
            new_values = util.Counter()
            
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                    continue
                
                action = self.policy.get(state)
                if action is None:
                    new_values[state] = 0
                    continue

                q_value = 0
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for next_state, prob in transitions:
                    reward = self.mdp.getReward(state, action, next_state)
                    q_value += prob * (reward + self.discount * self.values[next_state])
                
                new_values[state] = q_value
                max_change = max(max_change, abs(new_values[state] - self.values[state]))
            

            self.values = new_values
            
            if max_change < self.epsilon:
                break

    def policyImprovement(self):
        policy_changed = False
        
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
                
            best_action = self.computeActionFromValues(state)
            current_action = self.policy.get(state)
            
            if best_action is not None and best_action != current_action:
                self.policy[state] = best_action
                policy_changed = True
                
        return policy_changed

    def getValue(self, state):
        """Return the value of the state (computed in __init__)."""
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """Compute the Q-value of action in state from the value function stored in self.values."""
        value = 0.0

        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        
        for next_state, prob in transitions:
            reward = self.mdp.getReward(state, action, next_state)
            value += prob * (reward + self.discount * self.values[next_state])

        return value

    def computeActionFromValues(self, state):
        """The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        bestaction = None

        if self.mdp.isTerminal(state):
            return None
        
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        
        best_q_value = float('-inf')
        
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                bestaction = action

        return bestaction

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)