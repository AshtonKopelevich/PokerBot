import numpy as np

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from strategy.timing import TimeManager
from strategy.features import extract_features
from strategy.fsm import choose_fsm_action, size_raise
from strategy.monte_carlo import estimate_equity
from strategy.redraw import choose_redraw
from strategy.state_utils import get_legal_actions

class HybridPokerBot:
    def __init__(self):
        self.time_manager = TimeManager(total_match_seconds=180.0, reserve_seconds=12.0)
        self.rng = np.random.default_rng(20260328)
        self.hand_index = 0
        self.opponent_model = {
            "folds": 0,
            "calls": 0,
            "raises": 0,
            "checks": 0,
            "showdowns": 0,
        }

    def handle_new_round(self, game_state, round_state, active):
        self.hand_index += 1
        self.time_manager.start_hand()
 def handle_round_over(self, game_state, terminal_state, active):
        self.time_manager.end_hand()
        # Add opponent adaptation updates here if terminal/action history is exposed.

    def get_action(self, game_state, round_state, active):
        legal = get_legal_actions(round_state)
        features = extract_features(game_state, round_state, active, self.opponent_model)

        baseline = choose_fsm_action(features)
        baseline = self._sanitize_action(baseline, legal, features)

        # Fast-path: preflop is FSM only.
        if features["street"] == 0:
            redraw_action = choose_redraw(round_state, active, features, baseline)
            if redraw_action is not None:
                return self._sanitize_action(redraw_action, legal, features)
            return baseline

             uncertain = features["close_decision"] or features["continue_cost"] > 0
        budget = self.time_manager.decision_budget(features["street"], uncertain=uncertain)

        equity = estimate_equity(
            round_state=round_state,
            active=active,
            my_cards=features["my_cards"],
            board=features["board"],
            time_budget_sec=budget,
            rng=self.rng,
        )

        refined = self._equity_policy(features, legal, equity, baseline)

        # Redraw is combined with the chosen betting action.
        redraw_action = choose_redraw(round_state, active, features, refined)
        if redraw_action is not None:
            return self._sanitize_action(redraw_action, legal, features)

        return self._sanitize_action(refined, legal, features)

        def _equity_policy(self, features, legal, equity, baseline):
        ccost = features["continue_cost"]
        pot_odds = features["pot_odds"]
        street = features["street"]

        strong_value_threshold = 0.76 if street == 3 else 0.72
        thin_value_threshold = 0.62 if street == 3 else 0.58
        bluff_catch_margin = 0.04

        if ccost > 0:
            # Fold if clearly below pot odds.
            if equity + bluff_catch_margin < pot_odds and FoldAction in legal:
                return FoldAction()

            # Raise strong value.
            if equity >= strong_value_threshold and RaiseAction in legal:
                return RaiseAction(self._raise_amount(features, equity, polar=True))

            # Call if profitable enough.
            if equity >= pot_odds and CallAction in legal:
                return CallAction()

        if FoldAction in legal:
                return FoldAction()
            return baseline

        # No bet to us.
        if equity >= strong_value_threshold and RaiseAction in legal:
            return RaiseAction(self._raise_amount(features, equity, polar=True))

        if equity >= thin_value_threshold and RaiseAction in legal:
            return RaiseAction(self._raise_amount(features, equity, polar=False))

        if CheckAction in legal:
            return CheckAction()
        return baseline
        def _raise_amount(self, features, equity, polar=False):
        mn = features["min_raise"]
        mx = features["max_raise"]
        pot = features["pot_size"]
        if mx <= 0:
            return mn

        if polar:
            target = max(mn, min(mx, pot))
        else:
            target = max(mn, min(mx, max(2, pot // 2)))

        # Tiny bump for very strong hands.
        if equity >= 0.85:
            target = min(mx, max(target, (3 * pot) // 2 if pot > 0 else mn))

        return max(mn, min(mx, target))

    def _sanitize_action(self, action, legal, features):
    """
        Make sure returned action is legal.
        For RedrawAction, assume competition engine treats it as legal when redraw is available
        and the embedded action is legal. Adjust if your scaffold requires explicit RedrawAction checks.
        """
        action_type = type(action)

        if isinstance(action, RedrawAction):
            embedded = action.action
            if type(embedded) not in legal:
                embedded = self._fallback_non_redraw(legal, features)
                return RedrawAction(action.target_type, action.target_index, embedded)
            return action

        if action_type in legal:
            if isinstance(action, RaiseAction):
                amt = int(action.amount)
                mn = features["min_raise"]
                mx = features["max_raise"]
                amt = max(mn, min(mx, amt))
                return RaiseAction(amt)
            return action

        return self._fallback_non_redraw(legal, features)

    def _fallback_non_redraw(self, legal, features):
        if CheckAction in legal:
            return CheckAction()
        if CallAction in legal:
            return CallAction()
        if FoldAction in legal:
            return FoldAction()
        if RaiseAction in legal:
            return RaiseAction(size_raise(features, big=False))
        raise RuntimeError("No legal action available")