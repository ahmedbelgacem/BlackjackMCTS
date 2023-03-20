from envs.blackjack import MyBlackjackEnv
from algorithms.montecarlo import TreeSearch, TreeNode
from gym.envs.toy_text.blackjack import BlackjackEnv

env = MyBlackjackEnv()
tree = TreeSearch(env)

tree.search((10, 5, False), 10)
print(tree)

# game = BlackjackEnv(render_mode = 'human')

# state, _ = game.reset()

# terminated = False
# while not terminated:
#   tree.search(state, 10000)
#   _, action, _ = tree.get_action()
#   _, reward, terminated, _, _ = game.step(action)
# print(reward)
# game.close()