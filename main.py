from envs.blackjack import MyBlackjackEnv
from algorithms.montecarlo import Tree, TreeNode
from gym.envs.toy_text.blackjack import BlackjackEnv
import matplotlib.pyplot as plt
import networkx as nx

env = MyBlackjackEnv()
tree = Tree(env)

tree.search((2, 2, False), n_iters = 1000, n_rollouts = 1000)
tree.display()

# game = BlackjackEnv(render_mode = 'human')

# state, _ = game.reset()

# terminated = False
# while not terminated:
#   tree.search(state, 10000)
#   _, action, _ = tree.get_action()
#   _, reward, terminated, _, _ = game.step(action)
# print(reward)
# game.close()