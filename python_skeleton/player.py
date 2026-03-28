'''
Default entrypoint player for the python skeleton.
Strategy logic lives in skeleton/bot.py as documented in README.
'''

from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot


class Player(Bot):
    pass


if __name__ == '__main__':
    run_bot(Player(), parse_args())
