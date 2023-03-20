import math
from envs.blackjack import MyBlackjackEnv

colors = [
	'\033[31m',
	'\033[32m',
	'\033[33m',
	'\033[34m',
	'\033[35m',
	'\033[36m',
	'\033[37m',
	'\033[90m',
	'\033[91m',
	'\033[92m',
	'\033[93m',
	'\033[94m',
	'\033[95m',
	'\033[96m',
]
    
class TreeNode:
  def __init__(self, state: tuple, parent: 'TreeNode' = None) -> None:
    """Initialize a TreeNode for Monte Carlo Tree Search.

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
    self.is_terminal = state[0] >= 21
    
  @property
  def is_leaf(self) -> bool:
    """Indicates whether the node is a leaf node.
    Note: This is a property method because it needs to update on each call.

    Returns:
        True if node is a leaf node, else False.
    """        
    return not bool(self.children)
    
  def __str__(self, level = 0) -> str:
    ret = colors[level] + '\t'*level + 'Node {} - reward = {} - visits = {} - is{} leaf'.format(self.state, self.reward, self.visits, ' not'*(1 - self.is_leaf)) + '\n' + '\033[0m'
    for action, child in self.children:
        ret += child.__str__(level + 1)
    return ret

class TreeSearch:
  def __init__(self, env: MyBlackjackEnv, c: float = 1.4) -> None:
    """Initialize a Tree for Monte Carlo Tree Search.

    Arguments:
        env -- OpenAI Gym Blackjack environment

    Keyword Arguments:
        c -- Exploration constant (default: {1.4})
    """    
    self.env = env
    self.c = c
    
  def __str__(self) -> str:
    if hasattr(self, 'root'):
      return str(self.root)
    return 'Tree is empty!'
    
  def tree_policy(self, parent: TreeNode):
    """UCB score based policy for best child selection.

    Arguments:
        parent -- Parent node.
    """
    best_score = float('-inf')
    best_child = None
    for _, child in parent.children:
      # Calculate UCB score
      if child.visits:
        exploit = child.reward / child.visits
        explore = math.sqrt(math.log(parent.visits) / child.visits)
        score = exploit + self.c * explore
      else:
        score = float('inf')
      if score > best_score:
        best_score = score
        best_child = child
    return best_child
    
  def select(self, node: TreeNode):
    """Use tree policy to construct path from node to most promising leaf node.

    Arguments:
        node -- Root node.
    """
    # Traverse path down to leaf node (leaf node = does not have children)
    while not node.is_leaf:
      node = self.tree_policy(node)
    return node
  
  def expand(self, node: TreeNode):
    """Expand leaf node by creating a child node for each possible action.

    Arguments:
        node -- Node to expand.

    Raises:
        ValueError: Node is not a leaf and is already expanded.
    """    
    if not node.is_leaf:
      raise ValueError('Node is not a leaf and is already expanded.')
    
    if node.is_terminal:
      return
    for action in range(self.env.action_space.n): #! Maybe use env
      self.env.reset(init_observation = node.state)
      state, _, _, _, _ = self.env.step(action)
      child = TreeNode(state, parent = node)
      node.children.append((action, child))
      
  def simulate(self, node: TreeNode, n_sims: int = 1) -> float:
    """Run simulations starting from node on an OpenAI Gym environment.

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
    """Use the accumulated rewards of simulations to back up and update the values of nodes in the snowcap

    Arguments:
        node -- Starting node to backup from.
        reward -- Reward obtained from simulations
    """    
    while node is not None:
      node.visits += 1
      node.reward += reward
      node = node.parent
      
  def search(self, state: tuple, n_iters: int):
    """Run multiple iterations of selection, simuation, expansion, backup procedure

    Arguments:
        state -- Initial state to build from
        n_iters -- Number of iterations
    """    
    self.root = TreeNode(state)
    for _ in range(n_iters):
      # print('Iter [{}/{}]'.format(_, n_iters))
      # print(self, end = '')
      node = self.select(self.root)
      # print('Selected node is {} - visits = {}'.format(node.state, node.visits))
      if not node.visits: # First time traversing the node
        reward = self.simulate(node) # Rollout
        # print('Simulated reward = ', reward)
        self.backup(node, reward)
      else:
        self.expand(node)
      # print()
        
  def get_action(self):
    """Get best possible action from tree search

    Raises:
        AttributeError: Tree is empty

    Returns:
        State, action and reward of chosen path
    """    
    if not hasattr(self, 'root'):
      raise AttributeError('Tree is empty!')
    best_score = float('-inf')
    best_child = None
    best_action = None
    for action, child in self.root.children:
      # Calculate UCB score
      if child.visits:
        exploit = child.reward / child.visits
        explore = math.sqrt(math.log(self.root.visits) / child.visits)
        score = exploit + self.c * explore
      else:
        score = float('inf')
      if score > best_score:
        best_score = score
        best_child = child
        best_action = action
    return best_child, best_action, best_score