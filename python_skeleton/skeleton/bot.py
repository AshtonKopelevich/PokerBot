'''
Concrete bot implementation with a simple finite state machine by street.
'''

from .actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction


RANKS = '23456789TJQKA'


class Bot():
    '''
    A simple, robust pokerbot using street-based FSM decisions.
    '''

    def __init__(self):
        self.redraw_used_this_round = False
        self._cached_raise_bounds = (0, 0)

    def _rank_value(self, card):
        if not card or card == '??':
            return -1
        rank = card[0]
        return RANKS.index(rank) if rank in RANKS else -1

    def _is_suited(self, c1, c2):
        return len(c1) > 1 and len(c2) > 1 and c1[1] == c2[1]

    def _is_strong_preflop(self, cards):
        if len(cards) < 2:
            return False
        v0 = self._rank_value(cards[0])
        v1 = self._rank_value(cards[1])
        if v0 < 0 or v1 < 0:
            return False

        pair = cards[0][0] == cards[1][0]
        high = lambda v: v >= RANKS.index('Q')
        ace = RANKS.index('A')
        king = RANKS.index('K')
        queen = RANKS.index('Q')

        # Pair, AK, AQ, and broadway-suited combos are treated as strong opens.
        if pair:
            return True
        if (v0 == ace and v1 in (king, queen)) or (v1 == ace and v0 in (king, queen)):
            return True
        if high(v0) and high(v1) and self._is_suited(cards[0], cards[1]):
            return True
        return False

    def _lowest_hole_index(self, cards):
        if len(cards) < 2:
            return 0
        return 0 if self._rank_value(cards[0]) <= self._rank_value(cards[1]) else 1

    def _fallback_action(self, legal_actions):
        if CheckAction in legal_actions:
            return CheckAction()
        if FoldAction in legal_actions:
            return FoldAction()
        if CallAction in legal_actions:
            return CallAction()
        if RaiseAction in legal_actions:
            min_raise, _ = self._cached_raise_bounds
            return RaiseAction(min_raise)
        return FoldAction()

    def _validate_action(self, action, legal_actions, round_state):
        if isinstance(action, RedrawAction):
            if RedrawAction not in legal_actions:
                return False
            inner = action.action
            if type(inner) not in (set(legal_actions) - {RedrawAction}):
                return False
            if isinstance(inner, RaiseAction):
                min_raise, max_raise = round_state.raise_bounds()
                if not (min_raise <= inner.amount <= max_raise):
                    return False
            return True

        if type(action) not in legal_actions:
            return False
        if isinstance(action, RaiseAction):
            min_raise, max_raise = round_state.raise_bounds()
            if not (min_raise <= action.amount <= max_raise):
                return False
        return True

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.
        '''
        _ = game_state
        _ = round_state
        _ = active
        self.redraw_used_this_round = False

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.
        '''
        _ = game_state
        _ = terminal_state
        _ = active

    def get_action(self, game_state, round_state, active):
        '''
        Street-based FSM strategy with robust safety fallback.
        '''
        _ = game_state
        try:
            legal_actions = round_state.legal_actions()
            street = round_state.street
            my_cards = round_state.hands[active]
            my_pip = round_state.pips[active]
            opp_pip = round_state.pips[1 - active]
            continue_cost = opp_pip - my_pip

            # Cache to support fallback without repeated bound calls.
            self._cached_raise_bounds = (0, 0)
            if RaiseAction in legal_actions:
                self._cached_raise_bounds = round_state.raise_bounds()

            action = None

            # FSM state: Preflop
            if street == 0:
                is_strong = self._is_strong_preflop(my_cards)
                if is_strong:
                    if RaiseAction in legal_actions:
                        min_raise, _ = round_state.raise_bounds()
                        action = RaiseAction(min_raise)
                    elif CallAction in legal_actions:
                        action = CallAction()
                    elif CheckAction in legal_actions:
                        action = CheckAction()
                    else:
                        action = FoldAction()
                else:
                    if CheckAction in legal_actions:
                        action = CheckAction()
                    elif continue_cost <= 2 and CallAction in legal_actions:
                        action = CallAction()
                    else:
                        action = FoldAction()

            # FSM state: Flop
            elif street == 3:
                if CheckAction in legal_actions:
                    action = CheckAction()
                elif continue_cost <= 6 and CallAction in legal_actions:
                    action = CallAction()
                else:
                    action = FoldAction()

            # FSM state: Turn (includes requested redraw dummy logic)
            elif street == 4:
                facing_bet = continue_cost > 0
                can_redraw = (RedrawAction in legal_actions) and (not self.redraw_used_this_round)
                if can_redraw and facing_bet:
                    target_idx = self._lowest_hole_index(my_cards)
                    inner = CheckAction() if CheckAction in legal_actions else FoldAction()
                    action = RedrawAction('hole', target_idx, inner)
                    self.redraw_used_this_round = True
                elif CheckAction in legal_actions:
                    action = CheckAction()
                elif continue_cost <= 8 and CallAction in legal_actions:
                    action = CallAction()
                else:
                    action = FoldAction()

            # FSM state: River
            elif street == 5:
                if RaiseAction in legal_actions and continue_cost == 0 and self._is_strong_preflop(my_cards):
                    min_raise, _ = round_state.raise_bounds()
                    action = RaiseAction(min_raise)
                elif CheckAction in legal_actions:
                    action = CheckAction()
                elif continue_cost <= 6 and CallAction in legal_actions:
                    action = CallAction()
                else:
                    action = FoldAction()

            # Unknown street: safe default path.
            else:
                action = self._fallback_action(legal_actions)

            if not self._validate_action(action, legal_actions, round_state):
                return self._fallback_action(legal_actions)
            return action

        except Exception:
            legal_actions = round_state.legal_actions()
            return CheckAction() if CheckAction in legal_actions else FoldAction()
