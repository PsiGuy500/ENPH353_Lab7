import random
import pickle

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        try:
            with open(filename+".pickle", "rb") as f:
                self.q = pickle.load(f)
        except FileNotFoundError:
            print(f"File {filename}.pickle not found. Starting with empty Q-table.")
        print(f"Loaded file: {filename}.pickle")

    def saveQ(self, filename):
        with open(filename+".pickle", "wb") as f:
            pickle.dump(self.q, f)
        print(f"Wrote to file: {filename}.pickle")

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = {action: self.getQ(state, action) for action in self.actions}
            max_q = max(q_values.values())
            actions_with_max_q = [action for action, q in q_values.items() if q == max_q]
            action = random.choice(actions_with_max_q)

        if return_q:
            return action, self.getQ(state, action)
        return action

    def learn(self, state1, action1, reward, state2):
        current_q = self.getQ(state1, action1)
        max_q_new_state = max([self.getQ(state2, action) for action in self.actions])
        self.q[(state1, action1)] = current_q + self.alpha * (reward + self.gamma * max_q_new_state - current_q)
