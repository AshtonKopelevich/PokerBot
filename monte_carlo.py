import time
import numpy as np

from strategy.hand_eval import evaluate_7


RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = tuple(r + s for r in RANKS for s in SUITS)


def estimate_equity(round_state, active, my_cards, board, time_budget_sec, rng, samples_floor=32):
    """
    Heads-up equity estimation using NumPy RNG.
    Returns estimated showdown equity in [0, 1].
    """
    used = set(my_cards)
    used.update(board)
    deck = [c for c in FULL_DECK if c not in used]

    need_board = 5 - len(board)
    assert need_board >= 0

    wins = 0.0
    trials = 0
    start = time.perf_counter()

    n = len(deck)
    draw_count = 2 + need_board
# Always do at least a few samples.
    while trials < samples_floor or (time.perf_counter() - start < time_budget_sec):
        idx = rng.choice(n, size=draw_count, replace=False)
        opp = [deck[idx[0]], deck[idx[1]]]
        future = [deck[i] for i in idx[2:]]
        full_board = board + future

        hero_score = evaluate_7(my_cards + full_board)
        opp_score = evaluate_7(opp + full_board)

        if hero_score > opp_score:
            wins += 1.0
        elif hero_score == opp_score:
            wins += 0.5
        trials += 1

    return 0.5 if trials == 0 else wins / trials