from strategy.state_utils import (
    get_street,
    get_my_cards,
    get_board_cards,
    get_my_stack,
    get_opp_stack,
    get_continue_cost,
    get_pot_size,
    get_raise_bounds,
    get_legal_actions,
)

RANK_ORDER = "23456789TJQKA"


def extract_features(game_state, round_state, active, opponent_model):
    street = get_street(round_state)
    my_cards = get_my_cards(round_state, active)
    board = get_board_cards(round_state)
    my_stack = get_my_stack(round_state, active)
    opp_stack = get_opp_stack(round_state, active)
    continue_cost = get_continue_cost(round_state, active)
    pot_size = get_pot_size(round_state)
    min_raise, max_raise = get_raise_bounds(round_state, active)
    legal = get_legal_actions(round_state)

hand_code = encode_hole(my_cards)
    paired_board = has_pair_on_board(board)
    flush_draw_board = has_flush_pressure(board)
    straight_pressure = has_straight_pressure(board)
    spr = my_stack / max(1, pot_size)

    pot_odds = 0.0 if continue_cost <= 0 else continue_cost / max(1, pot_size + continue_cost)
    close_decision = continue_cost > 0 and 0.10 <= pot_odds <= 0.45

return {
        "street": street,
        "my_cards": my_cards,
        "board": board,
        "my_stack": my_stack,
        "opp_stack": opp_stack,
        "continue_cost": continue_cost,
        "pot_size": pot_size,
        "min_raise": min_raise,
        "max_raise": max_raise,
        "legal": legal,
        "spr": spr,
        "pot_odds": pot_odds,
        "close_decision": close_decision,
        "hand_code": hand_code,
        "paired_board": paired_board,
        "flush_draw_board": flush_draw_board,
        "straight_pressure": straight_pressure,
        "opponent_model": opponent_model,
    }

    def encode_hole(cards):
    c1, c2 = cards
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    i1 = RANK_ORDER.index(r1)
    i2 = RANK_ORDER.index(r2)
    if i2 > i1:
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def has_pair_on_board(board):
    ranks = [c[0] for c in board]
    return len(ranks) != len(set(ranks))

def has_flush_pressure(board):
    suits = [c[1] for c in board]
    for suit in "shdc":
        if suits.count(suit) >= 3:
            return True
    return False


def has_straight_pressure(board):
    if len(board) < 3:
        return False
    values = sorted(set(RANK_ORDER.index(c[0]) for c in board))
    for i in range(len(values) - 1):
        if values[i + 1] - values[i] <= 2:
            return True
    return False