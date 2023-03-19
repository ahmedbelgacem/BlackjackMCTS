import math
from envs.blackjack import MyBlackjackEnv

class TreeNode:
  def __init__(self, state: tuple, parent: 'TreeNode' = None) -> None:
    """_summary_

    Arguments:
        state -- A tuple composed of sum of cards in player's hand, the dealer's visit card and a boolean indicating whether there exists a usable ace in the player's hand.

    Keyword Arguments:
        parent -- Parent node (default: {None})
    """    
    self.state = state
    self.children = []
    self.parent = parent
    self.visits = 0
    self.reward = 0
    
  def __str__(self) -> str:
    return str(self.state)

class TreeSearch:
  def __init__(self, env: MyBlackjackEnv, c: float = 1.4) -> None:
    """_summary_

    Arguments:
        env -- OpenAI Gym Blackjack environment

    Keyword Arguments:
        c -- Exploration constant (default: {1.4})
    """    
    self.env = env
    self.c = c
    
  def tree_policy(self, parent: TreeNode):
    """UCB score based policy for best child selection 

    Arguments:
        parent -- Parent node
    """
    best_score = float('-inf')
    best_child = None
    for _, child in parent.children:
      # Calculate UCB score
      exploit = child.reward / child.visits
      explore = math.sqrt(math.log(parent.visits) / child.visits)
      score = exploit + self.c * explore
      if score > best_score:
        best_score = score
        best_child = child
    return best_child
    
  def select(self, root: TreeNode):
    """Use tree policy to construct path from root to most promising leaf node.

    Arguments:
        root -- Root node
    """
    # Cold start
    if not root.children:
      return root
    
    node = root
    # Traverse path down to leaf node (leaf node = does not have children)
    while node.children:
      node = self.tree_policy(node)
    return node
  
  def expand(self, node: TreeNode):
    """_summary_

    Arguments:
        node -- Node to expand

    Raises:
        ValueError: Node is not a leaf and is already expanded.
    """    
    if node.children:
      raise ValueError('node is not a leaf and is already expanded.')
    
    for action in [0, 1]:
      self.env.reset(init_observation = node.state)
      state, _, _, _, _ = self.env.step(action)
      child = TreeNode(state, parent = node)
      node.children.append((action, child))
      
  def simulate(self, node: TreeNode, n_sims: int = 1) -> float:
    """_summary_

    Arguments:
        node -- Node to run simulations on.

    Keyword Arguments:
        n_sims -- Number of simulations. Rewards are added up. (default: {1})

    Returns:
        Total reward.
    """    
    total_reward = 0
    for _ in range(n_sims):
      # Start from node
      self.env.reset(init_observation = node.state)
      # Simulate until temrination with random actions
      terminated  = False
      while not terminated:
        action = self.env.action_space.sample()
        _, reward, terminated, _, _ = self.env.step(action)
      total_reward += reward
      return total_reward
  
  def backup(self, node: TreeNode, reward: float):
    """Use the accumulated rewards of simulations to back up and update the values of nodes in the snowcap.

    Arguments:
        node -- Starting node to backup from.
        reward -- Reward obtained from simulations.
    """    
    while node is not None:
      node.visits += 1
      node.reward += reward
      node = node.parent
  
  # TODO
  def search(self, state: tuple, n_iters: int):
    """Execute a selection, expansion, simulation and backup round (not final).

    Arguments:
        state -- State to search from.
        n_iters -- Number of iterations.
    """    
    root = TreeNode(state)
    for _ in range(n_iters):
      leaf = self.select(root)
      if leaf.visits > 0:
        self.expand(leaf)
      reward = self.simulate(leaf)
      self.backup(leaf, reward)
    