from skeleton.actions import RedrawAction
from strategy.state_utils import redraw_available


RANK_ORDER = "23456789TJQKA"


def choose_redraw(round_state, active, features, embedded_action):
    """
    Heuristic redraw module.
    Conservative: only redraw when there is a clear likely gain.
    """
    if features["street"] >= 5:
        return None
    if not redraw_available(round_state, active):
        return None

    board = features["board"]
    my_cards = features["my_cards"]

    candidates = []
    # Hole redraw candidates.
    for i in range(2):
        score = score_hole_redraw(my_cards, board, i)
        candidates.append((score, "hole", i))

    # Board redraw candidates only after flop is out.
    for i in range(len(board)):
        score = score_board_redraw(my_cards, board, i)
        candidates.append((score, "board", i))

    best_score, target_type, target_index = max(candidates, key=lambda x: x[0])

    # Threshold tuned to be conservative.
    if best_score < 0.20:
        return None

    return RedrawAction(target_type, target_index, embedded_action)
    def score_hole_redraw(my_cards, board, idx):
    card = my_cards[idx]
    rank = RANK_ORDER.index(card[0])
    other = my_cards[1 - idx]

    score = 0.0

    # Low disconnected hole card.
    if rank <= 3:
        score += 0.16

    # Bad kicker when card does not connect to board.
    board_ranks = {c[0] for c in board}
    if board and card[0] not in board_ranks:
        score += 0.05
    # Break weak unsuited disconnected trash.
    if abs(RANK_ORDER.index(card[0]) - RANK_ORDER.index(other[0])) >= 4 and card[1] != other[1]:
        score += 0.04

    # Do not redraw high cards too eagerly.
    if rank >= 9:
        score -= 0.07

    return score


def score_board_redraw(my_cards, board, idx):
    target = board[idx]
    score = 0.0

    board_ranks = [c[0] for c in board]
    board_suits = [c[1] for c in board]
    my_ranks = {c[0] for c in my_cards}
    my_suits = {c[1] for c in my_cards}

    # Try to break scary paired boards if they do not help us.
    if board_ranks.count(target[0]) > 1 and target[0] not in my_ranks:
        score += 0.14

    # Try to break monotone / strong flush-pressure boards if they miss our suits.
    if board_suits.count(target[1]) >= 3 and target[1] not in my_suits:
        score += 0.12

    # Slightly favor removing a very high board rank that may improve villain broadway ranges.
    if RANK_ORDER.index(target[0]) >= 10 and target[0] not in my_ranks:
        score += 0.03

    return score