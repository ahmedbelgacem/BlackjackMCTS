from envs.blackjack import MyBlackjackEnv
from algorithms.montecarlo import TreeSearch, TreeNode

env = MyBlackjackEnv()
tree = TreeSearch(env)

root = TreeNode((11, 2, True))
tree.search((11, 2, True), 10)



