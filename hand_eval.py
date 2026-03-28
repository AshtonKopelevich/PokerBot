"""
Simple pure-Python 5-card / 7-card evaluator.
This is not the fastest possible evaluator, but it is correct enough for a strong sketch.
If needed later, this is the best place to optimize with numba/cython-compatible logic.
"""

RANK_ORDER = "23456789TJQKA"


def rank_value(card):
    return RANK_ORDER.index(card[0]) + 2

def evaluate_7(cards):
    assert len(cards) == 7
    best = None
    n = 7
    for a in range(n - 4):
        for b in range(a + 1, n - 3):
            for c in range(b + 1, n - 2):
                for d in range(c + 1, n - 1):
                    for e in range(d + 1, n):
                        value = evaluate_5([cards[a], cards[b], cards[c], cards[d], cards[e]])
                        if best is None or value > best:
                            best = value
    return best

def evaluate_5(cards):
    ranks = sorted((rank_value(c) for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    flush = len(set(suits)) == 1

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1

    unique_desc = sorted(counts.keys(), reverse=True)
    freq_sorted = sorted(((cnt, r) for r, cnt in counts.items()), reverse=True)

    straight, straight_high = straight_info(ranks)

    if straight and flush:
        return (8, straight_high)
 if freq_sorted[0][0] == 4:
        four = freq_sorted[0][1]
        kicker = max(r for r in unique_desc if r != four)
        return (7, four, kicker)

    if freq_sorted[0][0] == 3 and freq_sorted[1][0] == 2:
        return (6, freq_sorted[0][1], freq_sorted[1][1])

    if flush:
        return (5, *ranks)

    if straight:
        return (4, straight_high)

    if freq_sorted[0][0] == 3:
        trips = freq_sorted[0][1]
        kickers = sorted((r for r in unique_desc if r != trips), reverse=True)
        return (3, trips, *kickers)
    if freq_sorted[0][0] == 2 and freq_sorted[1][0] == 2:
        p1 = max(freq_sorted[0][1], freq_sorted[1][1])
        p2 = min(freq_sorted[0][1], freq_sorted[1][1])
        kicker = max(r for r in unique_desc if r not in (p1, p2))
        return (2, p1, p2, kicker)

    if freq_sorted[0][0] == 2:
        pair = freq_sorted[0][1]
        kickers = sorted((r for r in unique_desc if r != pair), reverse=True)
        return (1, pair, *kickers)

    return (0, *ranks)

    def straight_info(ranks_desc):
    uniq = sorted(set(ranks_desc))
    if len(uniq) != 5:
        return False, 0
    if uniq == [2, 3, 4, 5, 14]:
        return True, 5
    for i in range(4):
        if uniq[i + 1] != uniq[i] + 1:
            return False, 0
    return True, uniq[-1]