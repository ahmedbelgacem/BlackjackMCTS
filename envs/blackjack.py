# This acts mainly as an extension of OpenAI Gym's Blackjack environment to allow resetting at a specific observation. 
# This is important for the training of AI agents.

from gym.envs.toy_text import BlackjackEnv
from typing import Optional
import random

def generate_cards(observation: tuple) -> list:
  """Generate a random hand of cards for player using an observation (player_sum, dealer_card, is_usable_ace)

  Arguments:
      observation -- An observation according to OpenAI Gym Blackjack environment settings.

  Raises:
      ValueError: Observation should be valid.

  Returns:
      The player's hand of cards
  """  
  player_sum, _, is_usable_ace = observation
  hand = []
  current_sum = 0
  
  if player_sum < 11 and is_usable_ace:
    raise ValueError('Observation not valid.')
  
  if is_usable_ace:
    player_sum -= 10
  
  if is_usable_ace:
    hand.append(1)
    current_sum += 1
    while current_sum < player_sum:
      card = random.randint(1, min(10 , player_sum - current_sum))
      hand.append(card)
      current_sum = sum(hand)
  elif not is_usable_ace and player_sum < 11:
    while current_sum < player_sum:
      if min(10 , player_sum - current_sum) > 2:
        card = random.randint(2, min(10 , player_sum - current_sum))
      else:
        card = 2
      hand.append(card)
      current_sum = sum(hand)
  else:
    while current_sum < player_sum:
      card = random.randint(1, min(10 , player_sum - current_sum))
      hand.append(card)
      current_sum = sum(hand)
  return hand

class MyBlackjackEnv(BlackjackEnv):
  def __init__(self, render_mode: Optional[str] = None, natural = False, sab = False):
    super().__init__(render_mode = render_mode, natural = natural, sab = sab)
    
  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, init_observation: Optional[tuple] = None):
    if not init_observation:
      return super().reset(seed)
    
    _, dealer_card, _ = init_observation
    
    
    self.dealer = [dealer_card, random.randint(1, 10)]
    self.player = generate_cards(init_observation)
    
    _, dealer_card_value, _ = self._get_obs()
    suits = ['C', 'D', 'H', 'S']
    self.dealer_top_card_suit = self.np_random.choice(suits)

    if dealer_card_value == 1:
        self.dealer_top_card_value_str = 'A'
    elif dealer_card_value == 10:
        self.dealer_top_card_value_str = self.np_random.choice(['J', 'Q', 'K'])
    else:
        self.dealer_top_card_value_str = str(dealer_card_value)

    if self.render_mode == 'human':
        self.render()
    return self._get_obs(), {}