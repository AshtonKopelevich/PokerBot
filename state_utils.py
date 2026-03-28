"""
All scaffold-specific state access should be centralized here.
If your engine uses slightly different field names, update only this file.
"""


def get_street(round_state):
    return round_state.street


def get_my_cards(round_state, active):
    return list(round_state.hands[active])


def get_board_cards(round_state):
    street = round_state.street
    if street <= 0:
        return []
    return list(round_state.deck[:street])

def get_my_stack(round_state, active):
    return int(round_state.stacks[active])


def get_opp_stack(round_state, active):
    return int(round_state.stacks[1 - active])

def get_opp_stack(round_state, active):
    return int(round_state.stacks[1 - active])


def get_my_pip(round_state, active):
    return int(round_state.pips[active])


def get_opp_pip(round_state, active):
    return int(round_state.pips[1 - active])


def get_continue_cost(round_state, active):
    return get_opp_pip(round_state, active) - get_my_pip(round_state, active)

def get_pot_size(round_state):
    # Per-hand stacks reset to 400 each in this contest.
    return 800 - int(round_state.stacks[0]) - int(round_state.stacks[1])


def get_raise_bounds(round_state, active):
    if hasattr(round_state, "raise_bounds"):
        return round_state.raise_bounds()
    return (0, 0)

def get_legal_actions(round_state):
    return round_state.legal_actions()


def redraw_available(round_state, active):
    # Adapt this if the scaffold exposes redraw tracking differently.
    # Some scaffolds may expose a flag or a used-redraw count.
    # This sketch assumes either absence means available, or a boolean/array exists.
    if hasattr(round_state, "redraw_used"):
        used = round_state.redraw_used
        if isinstance(used, (list, tuple)):
            return not bool(used[active])
        return not bool(used)
    return round_state.street < 5