from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction


PREMIUM = {"AA", "KK", "QQ", "JJ", "TT", "AKs", "AQs", "AKo"}
STRONG = {
    "99", "88", "AQo", "AJs", "ATs", "KQs", "KJs", "QJs",
    "AJo", "KQo", "77", "66", "55", "44", "33", "22"
}
PLAYABLE = {
    "KTs", "QTs", "JTs", "T9s", "98s", "87s", "76s", "65s",
    "A9s", "A8s", "KJo", "QJo"
}

def choose_fsm_action(features):
    street = features["street"]
    if street == 0:
        return choose_preflop(features)
    return choose_postflop(features)


def choose_preflop(features):
    legal = features["legal"]
    hand_code = features["hand_code"]
    ccost = features["continue_cost"]

    if hand_code in PREMIUM:
        if RaiseAction in legal:
            return RaiseAction(size_raise(features, big=True))
        if CallAction in legal:
            return CallAction()
if hand_code in STRONG:
        if ccost == 0:
            if RaiseAction in legal:
                return RaiseAction(size_raise(features, big=False))
            if CheckAction in legal:
                return CheckAction()
        if ccost <= 8 and CallAction in legal:
            return CallAction()

if hand_code in PLAYABLE:
        if ccost == 0 and CheckAction in legal:
            return CheckAction()
        if ccost <= 4 and CallAction in legal:
            return CallAction()

    if ccost == 0 and CheckAction in legal:
        return CheckAction()
    if FoldAction in legal:
        return FoldAction()
    if CheckAction in legal:
        return CheckAction()
    return CallAction()

def choose_postflop(features):
    legal = features["legal"]
    ccost = features["continue_cost"]
    pot = features["pot_size"]

    if ccost == 0:
        if RaiseAction in legal and should_probe(features):
            return RaiseAction(size_raise(features, big=False))
        if CheckAction in legal:
            return CheckAction()

    cheap_continue = max(2, pot // 6)
    if ccost <= cheap_continue and CallAction in legal:
        return CallAction()

    if FoldAction in legal:
        return FoldAction()
    if CheckAction in legal:
        return CheckAction()
    return CallAction()

    def should_probe(features):
    street = features["street"]
    if street == 3 and not features["paired_board"] and not features["flush_draw_board"]:
        return True
    return False


def size_raise(features, big=False):
    mn = features["min_raise"]
    mx = features["max_raise"]
    pot = features["pot_size"]
    target = pot if big else max(2, pot // 2)
    if mx <= 0:
        return mn
    return max(mn, min(mx, target))