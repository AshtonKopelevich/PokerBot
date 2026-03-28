'''
Redraw-compatible example player for this repository's Hold'em + redraw rules.
'''

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction, RedrawAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


RANKS = '23456789TJQKA'


class Player(Bot):
    def _rank_value(self, card):
        if not card or card == '??':
            return -1
        return RANKS.index(card[0]) if card[0] in RANKS else -1

    def handle_new_round(self, game_state, round_state, active):
        _ = game_state
        _ = round_state
        _ = active

    def handle_round_over(self, game_state, terminal_state, active):
        _ = game_state
        _ = terminal_state
        _ = active

    def get_action(self, game_state, round_state, active):
        _ = game_state
        legal_actions = round_state.legal_actions()
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        continue_cost = opp_pip - my_pip

        # Lightweight redraw demonstration on flop/turn.
        if RedrawAction in legal_actions and round_state.street in (3, 4):
            my_cards = round_state.hands[active]
            idx = 0
            if len(my_cards) >= 2 and self._rank_value(my_cards[1]) < self._rank_value(my_cards[0]):
                idx = 1
            if CheckAction in legal_actions:
                return RedrawAction('hole', idx, CheckAction())
            if CallAction in legal_actions:
                return RedrawAction('hole', idx, CallAction())

        if RaiseAction in legal_actions and continue_cost == 0:
            min_raise, _ = round_state.raise_bounds()
            return RaiseAction(min_raise)
        if CheckAction in legal_actions:
            return CheckAction()
        if continue_cost <= 8 and CallAction in legal_actions:
            return CallAction()
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
