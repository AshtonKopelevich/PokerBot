from skeleton.bot import Bot
from strategy.bot_core import HybridPokerBot


class Player(Bot):
    def __init__(self):
        self.impl = HybridPokerBot()

    def handle_new_round(self, game_state, round_state, active):
        self.impl.handle_new_round(game_state, round_state, active)

    def handle_round_over(self, game_state, terminal_state, active):
        self.impl.handle_round_over(game_state, terminal_state, active)

    def get_action(self, game_state, round_state, active):
        return self.impl.get_action(game_state, round_state, active)