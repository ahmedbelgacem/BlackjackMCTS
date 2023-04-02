from envs.blackjack import MyBlackjackEnv
from algorithms.montecarlo import Tree

env = MyBlackjackEnv()
tree = Tree(env, c = 1.4)

tree.search((20, 10, False), n_iters = 10000, n_rollouts = 1000)
tree.display()