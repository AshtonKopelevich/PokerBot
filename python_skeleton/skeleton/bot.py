'''
This file contains the base class that you should implement for your pokerbot.
'''
from skeleton.decision_engine import DecisionEngine # Beautifully simple


class Bot():
    '''
    The base class for a pokerbot.
    '''
    def __init__(self):
        '''
        Called once when the bot is instantiated at the start of the match.
        Used to set up match-long tracking variables.
        '''
        # Match-level trackers
        self.match_clock = 180.0  # We have 180 seconds total
        self.hands_played = 0
        self.my_bankroll = 0      # Cumulative chip delta over the match
        
        # We'll use this later to track opponent tendencies (e.g., fold frequencies)
        self.opponent_profile = {
            'preflop_raises': 0,
            'folds_to_redraw': 0,
            'total_redraws_used': 0
        }

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        # 1. Update match-level stats
        self.hands_played = game_state.round_num
        self.match_clock = game_state.game_clock  # The engine tracks remaining time
        self.my_bankroll = game_state.bankroll    # Our net profit/loss so far
        
        # 2. Reset hand-level variables for the new hand
        # This is CRITICAL for the new 2026 redraw rule
        self.has_redrawn = False  
        
        # 3. Quick sanity check on time (logging for our own debugging)
        time_per_hand_left = self.match_clock / max(1, (300 - self.hands_played + 1))
        
        # If we're averaging less than 0.1 seconds left per hand, we might need 
        # to trigger a "panic mode" in get_action later to skip heavy math.
        self.time_panic = time_per_hand_left < 0.1

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        # 1. Calculate Bankroll Changes
        # terminal_state.deltas is a list containing the chip change for each player.
        my_delta = terminal_state.deltas[active]
        opponent_delta = terminal_state.deltas[1 - active]
        
        # Update our running bankroll (useful for logging/debugging)
        self.my_bankroll += my_delta
        
        # 2. Basic Opponent Profiling (The Foundation)
        # To truly exploit the opponent, we need to know what they are doing.
        # Here we can look back at the terminal_state to see if we reached showdown
        # and if they revealed their cards.
        
        # Example: Did we win or lose?
        if my_delta > 0:
            pass # We won!
        elif my_delta < 0:
            pass # We lost. 
            
        # Note: If you want to get advanced, you can parse terminal_state.previous_state
        # to see if the opponent folded to a raise, or if they used a RedrawAction 
        # and what card they discarded. That requires looping through the action history.

        # 3. Print a quick summary to the console for local testing
        # (Be sure to remove or minimize print statements in your final submission 
        # to save precious I/O milliseconds!)
        print(f"Hand {game_state.round_num} Over | Won/Lost: {my_delta} | Total Bankroll: {self.my_bankroll}")

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action (FoldAction, CallAction, CheckAction, RaiseAction, or RedrawAction).
        '''
        raise NotImplementedError('get_action')
