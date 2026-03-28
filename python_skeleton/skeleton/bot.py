'''
This file contains the base class that you should implement for your pokerbot.
'''


class Bot():
    '''
    The base class for a pokerbot.
    '''

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
         _ = game_state
        _ = round_state
        _ = active
        self.round_index += 1

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
        _ = game_state
        previous_state = terminal_state.previous_state
        opp_cards = previous_state.hands[1 - active]

        if opp_cards:
            self.opp_stats["showdowns"] += 1
            if self._three_card_strength(list(opp_cards)) >= 70:
                self.opp_stats["revealed_strong"] += 1

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
         legal = round_state.legal_actions()
        street = round_state.street

        my_cards = list(round_state.hands[active])
        board_cards = list(round_state.board)

        my_pip = int(round_state.pips[active])
        opp_pip = int(round_state.pips[1 - active])
        continue_cost = opp_pip - my_pip

        my_stack = int(round_state.stacks[active])
        opp_stack = int(round_state.stacks[1 - active])

        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        pot_size = my_contribution + opp_contribution

        min_raise, max_raise = (0, 0)
        if RaiseAction in legal:
            min_raise, max_raise = round_state.raise_bounds()
            min_raise = int(min_raise)
            max_raise = int(max_raise)

        self._observe_opponent(continue_cost, my_stack, pot_size, legal)

        opp_style = self._classify_opponent()

        # Mandatory discard phase
        if DiscardAction in legal:
            discard_idx = self._choose_discard_index(my_cards, board_cards, opp_style)
            return DiscardAction(discard_idx)

        # Preflop
        if street == 0:
            action = self._preflop_action(
                my_cards=my_cards,
                continue_cost=continue_cost,
                my_pip=my_pip,
                opp_pip=opp_pip,
                my_stack=my_stack,
                opp_stack=opp_stack,
                pot_size=pot_size,
                legal=legal,
                min_raise=min_raise,
                max_raise=max_raise,
                opp_style=opp_style,
            )
            return self._sanitize_action(action, legal, min_raise, max_raise)

        # Postflop / later streets
        budget = self._time_budget(street, facing_bet=(continue_cost > 0), opp_style=opp_style)

        equity = self._estimate_equity_variant(
            my_cards=my_cards,
            board_cards=board_cards,
            street=street,
            time_budget=budget,
            min_samples=16 if street < 5 else 24,
        )

        action = self._postflop_action(
            my_cards=my_cards,
            board_cards=board_cards,
            street=street,
            equity=equity,
            continue_cost=continue_cost,
            pot_size=pot_size,
            my_pip=my_pip,
            opp_pip=opp_pip,
            my_stack=my_stack,
            opp_stack=opp_stack,
            legal=legal,
            min_raise=min_raise,
            max_raise=max_raise,
            opp_style=opp_style,
        )
        return self._sanitize_action(action, legal, min_raise, max_raise)
        raise NotImplementedError('get_action')
