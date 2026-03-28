import numpy as np
import json
import os
import time

class HandEvaluator:
    def __init__(self):
        # 1. Load Preflop Equities from JSON
        # Get the absolute path to the directory where THIS file lives
        current_dir = os.path.dirname(__file__)
        json_path = os.path.join(current_dir, 'holdem_probabilities.json')
        
        try:
            with open(json_path, 'r') as file:
                self.preflop_equities = json.load(file)
        except FileNotFoundError:
            print(f"CRITICAL ERROR: {json_path} not found!")
            # Fallback to prevent immediate crashing, though the bot will play blind preflop
            self.preflop_equities = {'AAp': 0.85, '72o': 0.32} 
            
        # 2. Card Mappings (Strings to Integers 0-51)
        # Rank: 0-12 (2 to Ace). Suit: 0-3. Int = Rank + (Suit * 13)
        self.rank_map = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, 
                         '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
        self.suit_map = {'s':0, 'h':1, 'd':2, 'c':3}


    # --- MAIN INTERFACES ---

    def get_baseline_equity(self, my_cards_str, board_cards_str, iterations=1000):
        """Routes to preflop lookup or postflop Monte Carlo."""
        if not board_cards_str:
            return self._get_preflop_equity(my_cards_str)
        return self._run_monte_carlo(my_cards_str, board_cards_str, iterations)

    def evaluate_all_redraws(self, my_cards_str, board_cards_str, iterations=500):
        """Simulates the EV of redrawing every possible valid card."""
        redraw_equities = {}
        
        # 1. Simulate Hole Card Redraws
        for i in range(len(my_cards_str)):
            kept_hole = [my_cards_str[1 - i]]
            eq = self._run_monte_carlo(kept_hole, board_cards_str, iterations, draw_type='hole')
            redraw_equities[f'hole_{i}'] = eq
            
        # 2. Simulate Board Card Redraws
        for i in range(len(board_cards_str)):
            kept_board = board_cards_str[:i] + board_cards_str[i+1:]
            eq = self._run_monte_carlo(my_cards_str, kept_board, iterations, draw_type='board')
            redraw_equities[f'board_{i}'] = eq
            
        return redraw_equities

    # --- INTERNAL LOGIC & PARSING ---

    def _card_to_int(self, card_str):
        return self.rank_map[card_str[0]] + (self.suit_map[card_str[1]] * 13)

    def _get_preflop_equity(self, my_cards_str):
        """Parses ['Ah', 'Kd'] into 'AKo' and looks up baseline equity."""
        r1, s1 = my_cards_str[0][0], my_cards_str[0][1]
        r2, s2 = my_cards_str[1][0], my_cards_str[1][1]
        
        val1, val2 = self.rank_map[r1], self.rank_map[r2]
        if val1 < val2:
            r1, r2 = r2, r1 # Ensure highest card is first
            
        if r1 == r2:
            key = f"{r1}{r2}p"
        else:
            suffix = 's' if s1 == s2 else 'o'
            key = f"{r1}{r2}{suffix}"
            
        # Default to 0.5 if you haven't filled out the whole table yet
        return self.preflop_equities.get(key, 0.5)

    # --- MONTE CARLO ENGINE ---

    def _run_monte_carlo(self, my_cards_str, board_cards_str, iterations, draw_type=None):
        """
        A vectorized Monte Carlo simulator using numpy.
        draw_type: None, 'hole', or 'board'. If set, it draws 1 extra card first.
        """
        # Convert known strings to ints
        my_cards = [self._card_to_int(c) for c in my_cards_str]
        board_cards = [self._card_to_int(c) for c in board_cards_str]
        known_cards = set(my_cards + board_cards)
        
        # Build the remaining deck
        deck = np.array([c for c in range(52) if c not in known_cards], dtype=np.int8)
        
        # Calculate how many cards we need to pull from the deck per simulation
        cards_needed_for_board = 5 - len(board_cards)
        cards_needed_for_opp = 2
        cards_needed_for_redraw = 1 if draw_type else 0
        
        total_draws = cards_needed_for_board + cards_needed_for_opp + cards_needed_for_redraw
        
        # Create a matrix of random deck draws for ALL iterations simultaneously
        # np.argsort on random numbers is a blazing fast way to shuffle in 2D
        random_indices = np.argsort(np.random.rand(iterations, len(deck)), axis=1)[:, :total_draws]
        drawn_matrices = deck[random_indices]
        
        wins = 0
        ties = 0
        
        # Slice the drawn matrix to assign cards to their respective spots
        col_idx = 0
        
        if draw_type == 'hole':
            simulated_hole_draws = drawn_matrices[:, col_idx:col_idx+1]
            col_idx += 1
        elif draw_type == 'board':
            simulated_board_draws = drawn_matrices[:, col_idx:col_idx+1]
            col_idx += 1
            
        opp_hole_draws = drawn_matrices[:, col_idx:col_idx+2]
        col_idx += 2
        
        if cards_needed_for_board > 0:
            board_completion_draws = drawn_matrices[:, col_idx:]
            
        # Run the evaluation loop
        # Note: To squeeze this into 0.6 seconds, we do a fast loop. 
        # For maximum performance, wrap the logic below in @numba.njit
        for i in range(iterations):
            # 1. Reconstruct My Hand
            current_my_cards = list(my_cards)
            if draw_type == 'hole':
                current_my_cards.append(simulated_hole_draws[i][0])
                
            # 2. Reconstruct Board
            current_board = list(board_cards)
            if draw_type == 'board':
                current_board.append(simulated_board_draws[i][0])
            if cards_needed_for_board > 0:
                current_board.extend(board_completion_draws[i])
                
            # 3. Reconstruct Opponent Hand
            current_opp_cards = list(opp_hole_draws[i])
            
            # 4. Evaluate (Requires a scoring function)
            my_score = self._score_7_card_hand(current_my_cards + current_board)
            opp_score = self._score_7_card_hand(current_opp_cards + current_board)
            
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
                
        # Return Equity (Wins + Half of Ties)
        return (wins + (ties / 2.0)) / iterations

    # --- HAND SCORER ---

    def _score_7_card_hand(self, cards):
        """
        A placeholder heuristic scorer. 
        In production, replace this with a perfect hash lookup or the pkrbot evaluator
        if the engine provides one, because stringing if/else statements is the 
        biggest bottleneck in Monte Carlo.
        Returns an integer score (higher = better hand).
        """
        # Quick and dirty rank array (0-12) and suit array (0-3)
        ranks = sorted([c % 13 for c in cards], reverse=True)
        suits = [c // 13 for c in cards]
        
        # --- Simplified evaluation logic for the sake of the engine running ---
        # This gives a rough hierarchical score. 
        # (A true evaluator checks flushes and straights meticulously).
        
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        counts = sorted(rank_counts.values(), reverse=True)
        
        score = 0
        if counts[0] == 4:
            score = 8000000 + (ranks[0] * 10000) # Quads
        elif counts[0] == 3 and (len(counts) > 1 and counts[1] >= 2):
            score = 7000000 + (ranks[0] * 10000) # Full House
        elif counts[0] == 3:
            score = 4000000 + (ranks[0] * 10000) # Trips
        elif counts[0] == 2 and (len(counts) > 1 and counts[1] == 2):
            score = 3000000 + (ranks[0] * 10000) # Two Pair
        elif counts[0] == 2:
            score = 2000000 + (ranks[0] * 10000) # Pair
        else:
            score = 1000000 + (ranks[0] * 10000) # High Card
            
        # Add kicker values roughly
        score += sum([r * (10 ** i) for i, r in enumerate(reversed(ranks[:5]))])
        return score