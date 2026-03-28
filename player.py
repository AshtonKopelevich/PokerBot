"""
Adaptive single-file bot for the 3-hole-card + discard variant.

Targeted to outperform:
- all-in / max-pressure bots
- passive check/call bots
- simple baseline bots

Notes:
- Uses only numpy + provided skeleton
- Adapts opponent model across the match
- Discard logic is heuristic but reasonably strong
- Lightweight Monte Carlo is used on later betting streets
"""

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction, DiscardAction
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from skeleton.states import STARTING_STACK
import time
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


class Player(Bot):
    def __init__(self):
        self.rng = np.random.default_rng(20260328)

        self.match_start = time.perf_counter()
        self.total_time_limit = 180.0
        self.reserve_time = 12.0

        self.round_index = 0

        # Opponent model
        self.opp_stats = {
            "faced_actions": 0,
            "faced_big_bets": 0,
            "faced_huge_bets": 0,
            "faced_small_bets": 0,
            "free_checks_seen": 0,
            "showdowns": 0,
            "revealed_strong": 0,
        }

    # =========================================================
    # Required methods
    # =========================================================

    def handle_new_round(self, game_state, round_state, active):
        _ = game_state
        _ = round_state
        _ = active
        self.round_index += 1

    def handle_round_over(self, game_state, terminal_state, active):
        _ = game_state
        previous_state = terminal_state.previous_state
        opp_cards = previous_state.hands[1 - active]

        if opp_cards:
            self.opp_stats["showdowns"] += 1
            if self._three_card_strength(list(opp_cards)) >= 70:
                self.opp_stats["revealed_strong"] += 1

    def get_action(self, game_state, round_state, active):
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

    # =========================================================
    # Opponent model
    # =========================================================

    def _observe_opponent(self, continue_cost, my_stack, pot_size, legal):
        if continue_cost > 0:
            self.opp_stats["faced_actions"] += 1
            if continue_cost >= max(40, my_stack // 3):
                self.opp_stats["faced_huge_bets"] += 1
            elif continue_cost >= max(12, pot_size // 2):
                self.opp_stats["faced_big_bets"] += 1
            else:
                self.opp_stats["faced_small_bets"] += 1
        else:
            if CheckAction in legal:
                self.opp_stats["free_checks_seen"] += 1

    def _classify_opponent(self):
        faced = max(1, self.opp_stats["faced_actions"])
        huge_rate = self.opp_stats["faced_huge_bets"] / faced
        big_rate = self.opp_stats["faced_big_bets"] / faced
        small_rate = self.opp_stats["faced_small_bets"] / faced
        free_rate = self.opp_stats["free_checks_seen"] / max(1, self.round_index)

        if huge_rate >= 0.35 or (huge_rate + big_rate) >= 0.65:
            return "maniac"

        if free_rate >= 0.50 and huge_rate <= 0.10 and small_rate >= 0.30:
            return "passive"

        return "normal"

    # =========================================================
    # Time management
    # =========================================================

    def _time_budget(self, street, facing_bet, opp_style):
        used = time.perf_counter() - self.match_start
        remaining = max(0.0, self.total_time_limit - used)
        usable = max(0.0, remaining - self.reserve_time)

        if usable <= 0.0:
            return 0.001

        if street <= 3:
            base = 0.004 if facing_bet else 0.002
        elif street == 4:
            base = 0.006 if facing_bet else 0.003
        elif street == 5:
            base = 0.008 if facing_bet else 0.004
        else:
            base = 0.010 if facing_bet else 0.005

        if opp_style == "maniac" and facing_bet:
            base *= 1.25

        return min(base, usable)

    # =========================================================
    # Preflop
    # =========================================================

    def _preflop_action(
        self,
        my_cards,
        continue_cost,
        my_pip,
        opp_pip,
        my_stack,
        opp_stack,
        pot_size,
        legal,
        min_raise,
        max_raise,
        opp_style,
    ):
        strength = self._three_card_strength(my_cards)
        best_two = self._best_two_code_from_three(my_cards)

        if opp_style == "maniac":
            return self._preflop_vs_maniac(
                strength, best_two, continue_cost, legal, min_raise, max_raise, my_pip, opp_pip
            )

        if opp_style == "passive":
            return self._preflop_vs_passive(
                strength, best_two, continue_cost, legal, min_raise, max_raise, my_pip, opp_pip
            )

        return self._preflop_vs_normal(
            strength, best_two, continue_cost, legal, min_raise, max_raise, my_pip, opp_pip
        )

    def _preflop_vs_maniac(self, strength, best_two, continue_cost, legal, min_raise, max_raise, my_pip, opp_pip):
        # Tight value-heavy preflop versus all-in style
        if continue_cost > 0:
            if strength >= 78 and RaiseAction in legal:
                return RaiseAction(max_raise)
            if strength >= 62 and CallAction in legal:
                return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return CheckAction() if CheckAction in legal else CallAction()

        if RaiseAction in legal and strength >= 76:
            return RaiseAction(max_raise)

        if CheckAction in legal:
            return CheckAction()

        if CallAction in legal and strength >= 50:
            return CallAction()

        return FoldAction() if FoldAction in legal else CheckAction()

    def _preflop_vs_passive(self, strength, best_two, continue_cost, legal, min_raise, max_raise, my_pip, opp_pip):
        # Steal more because passive bot checks/folds too much
        if continue_cost > 0:
            if strength >= 58 and CallAction in legal:
                return CallAction()
            if strength >= 78 and RaiseAction in legal:
                return RaiseAction(self._pressure_raise_total(opp_pip, min_raise, max_raise, desired_continue=12))
            if FoldAction in legal:
                return FoldAction()
            return CheckAction() if CheckAction in legal else CallAction()

        if RaiseAction in legal:
            if strength >= 45 or self._is_steal_candidate(best_two):
                return RaiseAction(self._pressure_raise_total(opp_pip, min_raise, max_raise, desired_continue=12))

        if CheckAction in legal:
            return CheckAction()

        if CallAction in legal and strength >= 40:
            return CallAction()

        return FoldAction() if FoldAction in legal else CheckAction()

    def _preflop_vs_normal(self, strength, best_two, continue_cost, legal, min_raise, max_raise, my_pip, opp_pip):
        if continue_cost > 0:
            if strength >= 72 and RaiseAction in legal:
                return RaiseAction(min(max_raise, max(min_raise, opp_pip + 18)))
            if strength >= 54 and CallAction in legal:
                return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return CheckAction() if CheckAction in legal else CallAction()

        if RaiseAction in legal and strength >= 55:
            return RaiseAction(min(max_raise, max(min_raise, opp_pip + 12)))

        if CheckAction in legal:
            return CheckAction()

        if CallAction in legal and strength >= 45:
            return CallAction()

        return FoldAction() if FoldAction in legal else CheckAction()

    # =========================================================
    # Postflop
    # =========================================================

    def _postflop_action(
        self,
        my_cards,
        board_cards,
        street,
        equity,
        continue_cost,
        pot_size,
        my_pip,
        opp_pip,
        my_stack,
        opp_stack,
        legal,
        min_raise,
        max_raise,
        opp_style,
    ):
        hand_cat = self._current_category(my_cards, board_cards)

        if continue_cost > 0:
            pot_odds = continue_cost / max(1, pot_size + continue_cost)

            if opp_style == "maniac":
                if equity >= max(0.54, pot_odds + 0.04):
                    if RaiseAction in legal and equity >= 0.76:
                        return RaiseAction(max_raise)
                    if CallAction in legal:
                        return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CheckAction() if CheckAction in legal else CallAction()

            if opp_style == "passive":
                # Passive aggression is stronger on average
                if hand_cat >= 2 and RaiseAction in legal:
                    return RaiseAction(self._value_raise_total(opp_pip, pot_size, min_raise, max_raise))
                if equity >= max(0.62, pot_odds + 0.08) and CallAction in legal:
                    return CallAction()
                if FoldAction in legal:
                    return FoldAction()
                return CheckAction() if CheckAction in legal else CallAction()

            # Normal
            if hand_cat >= 2 and RaiseAction in legal and equity >= 0.68:
                return RaiseAction(self._value_raise_total(opp_pip, pot_size, min_raise, max_raise))
            if equity >= max(0.57, pot_odds + 0.05) and CallAction in legal:
                return CallAction()
            if FoldAction in legal:
                return FoldAction()
            return CheckAction() if CheckAction in legal else CallAction()

        # No bet to us
        if opp_style == "maniac":
            # Trap/check more often unless strong
            if equity >= 0.78 and RaiseAction in legal:
                return RaiseAction(self._value_raise_total(opp_pip, pot_size, min_raise, max_raise))
            if CheckAction in legal:
                return CheckAction()
            return CallAction() if CallAction in legal else FoldAction()

        if opp_style == "passive":
            # Punish checks with frequent pressure
            if RaiseAction in legal:
                if hand_cat >= 1 or equity >= 0.60 or self._good_stab_board(my_cards, board_cards):
                    return RaiseAction(self._pressure_raise_total(opp_pip, min_raise, max_raise, desired_continue=12))
            if CheckAction in legal:
                return CheckAction()
            return CallAction() if CallAction in legal else FoldAction()

        # Normal
        if RaiseAction in legal and (hand_cat >= 1 or equity >= 0.67):
            return RaiseAction(self._value_raise_total(opp_pip, pot_size, min_raise, max_raise))
        if RaiseAction in legal and street in (2, 3, 4) and self._good_stab_board(my_cards, board_cards) and equity >= 0.48:
            return RaiseAction(self._pressure_raise_total(opp_pip, min_raise, max_raise, desired_continue=10))
        if CheckAction in legal:
            return CheckAction()
        return CallAction() if CallAction in legal else FoldAction()

    # =========================================================
    # Discard logic
    # =========================================================

    def _choose_discard_index(self, my_cards, board_cards, opp_style):
        # Keep trips/pairs if possible
        ranks = [c[0] for c in my_cards]
        if len(set(ranks)) == 1:
            return 0  # arbitrary; extremely rare, all three same rank impossible in standard deck? kept for safety

        # Score each card for how much we want to KEEP it
        keep_scores = []
        board_ranks = {c[0] for c in board_cards}
        board_suits = [c[1] for c in board_cards]

        for i, card in enumerate(my_cards):
            score = 0.0
            rank_idx = RANKS.index(card[0])

            # High card value
            score += 0.6 * rank_idx

            # Pairing with our hand
            for j, other in enumerate(my_cards):
                if i != j and other[0] == card[0]:
                    score += 5.0

            # Pairing board
            if card[0] in board_ranks:
                score += 3.0

            # Suit coordination
            if board_suits.count(card[1]) >= 2:
                score += 1.2

            # Connectivity with other hole cards
            for j, other in enumerate(my_cards):
                if i != j:
                    gap = abs(RANKS.index(other[0]) - rank_idx)
                    if gap <= 1:
                        score += 1.4
                    elif gap == 2:
                        score += 0.7

            keep_scores.append(score)

        # Against passive bots, discard lowest-value card more aggressively
        # Against maniacs, keep raw showdown strength a bit more
        if opp_style == "maniac":
            pass
        elif opp_style == "passive":
            for i, card in enumerate(my_cards):
                if RANKS.index(card[0]) <= RANKS.index("7"):
                    keep_scores[i] -= 0.4

        # Discard the card with smallest keep score
        weakest = 0
        for i in range(1, len(keep_scores)):
            if keep_scores[i] < keep_scores[weakest]:
                weakest = i
        return weakest

    # =========================================================
    # Monte Carlo
    # =========================================================

    def _estimate_equity_variant(self, my_cards, board_cards, street, time_budget, min_samples=16):
        """
        Lightweight equity estimate for this discard variant.

        Approximation:
        - If holding 3 cards, we estimate by allowing our hand evaluator to choose the best 2-of-3
          at showdown.
        - Opponent is sampled with same number of hole cards as we currently have.
        - Future board is filled to 6 cards.
        """
        used = set(my_cards + board_cards)
        deck = [c for c in FULL_DECK if c not in used]

        opp_hole_count = len(my_cards)
        need_board = max(0, 6 - len(board_cards))
        draw_count = opp_hole_count + need_board

        wins = 0.0
        trials = 0
        start = time.perf_counter()
        deck_len = len(deck)

        while trials < min_samples or (time.perf_counter() - start < time_budget):
            idx = self.rng.choice(deck_len, size=draw_count, replace=False)

            opp_cards = [deck[int(idx[i])] for i in range(opp_hole_count)]
            future_board = [deck[int(idx[i])] for i in range(opp_hole_count, draw_count)]
            full_board = board_cards + future_board

            my_score = self._evaluate_variant_showdown(my_cards, full_board)
            opp_score = self._evaluate_variant_showdown(opp_cards, full_board)

            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5

            trials += 1

        return 0.5 if trials == 0 else wins / trials

    # =========================================================
    # Hand evaluation for variant
    # =========================================================

    def _evaluate_variant_showdown(self, hole_cards, board_cards):
        """
        At showdown in this variant:
        - final hand uses remaining 2 hole cards
        - there are 6 board cards
        - choose best 5-card poker hand from available cards

        If still holding 3 cards in our approximation, choose the best 2 hole cards to keep.
        """
        if len(hole_cards) <= 2:
            return self._best5_from_cards(hole_cards + board_cards)

        best = None
        n = len(hole_cards)
        for i in range(n):
            kept = [hole_cards[j] for j in range(n) if j != i]  # discard one hole card
            score = self._best5_from_cards(kept + board_cards)
            if best is None or score > best:
                best = score
        return best

    def _best5_from_cards(self, cards):
        best = None
        n = len(cards)
        for a in range(n - 4):
            for b in range(a + 1, n - 3):
                for c in range(b + 1, n - 2):
                    for d in range(c + 1, n - 1):
                        for e in range(d + 1, n):
                            score = self._evaluate_5([cards[a], cards[b], cards[c], cards[d], cards[e]])
                            if best is None or score > best:
                                best = score
        return best

    def _current_category(self, my_cards, board_cards):
        total = list(my_cards) + list(board_cards)
        if len(total) < 5:
            return 0
        return self._evaluate_variant_showdown(my_cards, board_cards)[0]

    def _evaluate_5(self, cards):
        ranks = sorted([RANKS.index(c[0]) + 2 for c in cards], reverse=True)
        suits = [c[1] for c in cards]
        flush = len(set(suits)) == 1

        counts = {}
        for r in ranks:
            counts[r] = counts.get(r, 0) + 1

        unique_ranks = sorted(counts.keys(), reverse=True)
        freq = sorted([(cnt, rank) for rank, cnt in counts.items()], reverse=True)

        straight, straight_high = self._straight_info(ranks)

        if straight and flush:
            return (8, straight_high)

        if freq[0][0] == 4:
            four = freq[0][1]
            kicker = max(r for r in unique_ranks if r != four)
            return (7, four, kicker)

        if freq[0][0] == 3 and freq[1][0] == 2:
            return (6, freq[0][1], freq[1][1])

        if flush:
            return (5, *ranks)

        if straight:
            return (4, straight_high)

        if freq[0][0] == 3:
            trips = freq[0][1]
            kickers = sorted([r for r in unique_ranks if r != trips], reverse=True)
            return (3, trips, *kickers)

        if freq[0][0] == 2 and freq[1][0] == 2:
            p1 = max(freq[0][1], freq[1][1])
            p2 = min(freq[0][1], freq[1][1])
            kicker = max(r for r in unique_ranks if r not in (p1, p2))
            return (2, p1, p2, kicker)

        if freq[0][0] == 2:
            pair = freq[0][1]
            kickers = sorted([r for r in unique_ranks if r != pair], reverse=True)
            return (1, pair, *kickers)

        return (0, *ranks)

    def _straight_info(self, ranks_desc):
        uniq = sorted(set(ranks_desc))
        if len(uniq) != 5:
            return False, 0
        if uniq == [2, 3, 4, 5, 14]:
            return True, 5
        for i in range(4):
            if uniq[i + 1] != uniq[i] + 1:
                return False, 0
        return True, uniq[-1]

    # =========================================================
    # Hand strength helpers
    # =========================================================

    def _three_card_strength(self, cards):
        """
        Rough preflop strength for 3-card starting hand.
        """
        ranks = sorted([RANKS.index(c[0]) for c in cards], reverse=True)
        suits = [c[1] for c in cards]

        score = 0.0

        # Raw high-card value
        score += 4.0 * ranks[0] + 2.5 * ranks[1] + 1.2 * ranks[2]

        # Pairs / trips
        if cards[0][0] == cards[1][0] == cards[2][0]:
            score += 40.0
        else:
            counts = {}
            for c in cards:
                counts[c[0]] = counts.get(c[0], 0) + 1
            if 2 in counts.values():
                score += 18.0

        # Suit value
        max_suit_count = max(suits.count(s) for s in SUITS)
        if max_suit_count == 3:
            score += 7.0
        elif max_suit_count == 2:
            score += 3.0

        # Connectivity
        gaps = sorted(ranks, reverse=True)
        diff1 = gaps[0] - gaps[1]
        diff2 = gaps[1] - gaps[2]
        if diff1 <= 1:
            score += 4.0
        elif diff1 == 2:
            score += 2.0
        if diff2 <= 1:
            score += 4.0
        elif diff2 == 2:
            score += 2.0

        # Aces and broadway boost
        ace_count = sum(1 for c in cards if c[0] == "A")
        score += 6.0 * ace_count
        broadway_count = sum(1 for c in cards if c[0] in "TJQKA")
        score += 1.5 * broadway_count

        # Normalize roughly to 0..100
        return min(100.0, max(0.0, score))

    def _best_two_code_from_three(self, cards):
        best = None
        best_score = None
        for i in range(3):
            for j in range(i + 1, 3):
                code = self._encode_two([cards[i], cards[j]])
                score = self._two_card_score(code)
                if best_score is None or score > best_score:
                    best_score = score
                    best = code
        return best

    def _two_card_score(self, code):
        if len(code) == 2 and code[0] == code[1]:
            return 100 + RANKS.index(code[0])

        hi = RANKS.index(code[0])
        lo = RANKS.index(code[1])
        suited = code.endswith("s")

        score = 4 * hi + 2 * lo
        if suited:
            score += 3
        if hi == RANKS.index("A"):
            score += 8
        if abs(hi - lo) <= 1:
            score += 4
        elif abs(hi - lo) == 2:
            score += 2
        return score

    def _is_steal_candidate(self, code):
        if len(code) == 2 and code[0] == code[1]:
            return True
        if code[0] == "A":
            return True
        return code in {
            "KQs", "KJs", "KTs", "KQo",
            "QJs", "QTs", "QJo",
            "JTs", "T9s", "98s", "87s", "76s"
        }

    # =========================================================
    # Raise sizing
    # =========================================================

    def _pressure_raise_total(self, opp_pip, min_raise, max_raise, desired_continue=12):
        target = opp_pip + desired_continue
        target = max(target, min_raise)
        if max_raise > 0:
            target = min(target, max_raise)
        return max(min_raise, target)

    def _value_raise_total(self, opp_pip, pot_size, min_raise, max_raise):
        target = max(opp_pip + 14, pot_size // 2)
        target = max(target, min_raise)
        if max_raise > 0:
            target = min(target, max_raise)
        return max(min_raise, target)

    # =========================================================
    # Board texture
    # =========================================================

    def _good_stab_board(self, my_cards, board_cards):
        if len(board_cards) < 2:
            return False

        board_ranks = [RANKS.index(c[0]) for c in board_cards]
        my_ranks = [RANKS.index(c[0]) for c in my_cards]

        paired_board = len({c[0] for c in board_cards}) < len(board_cards)
        if paired_board:
            return False

        if max(my_ranks) > max(board_ranks):
            return True

        combined = sorted(set(board_ranks + my_ranks))
        local_gaps = 0
        for i in range(len(combined) - 1):
            local_gaps += combined[i + 1] - combined[i]
        return local_gaps <= 8

    # =========================================================
    # Action sanitation
    # =========================================================

    def _sanitize_action(self, action, legal, min_raise, max_raise):
        if type(action) in legal:
            if isinstance(action, RaiseAction):
                amount = int(action.amount)
                amount = max(min_raise, min(max_raise, amount))
                return RaiseAction(amount)
            return action

        if CheckAction in legal:
            return CheckAction()
        if CallAction in legal:
            return CallAction()
        if FoldAction in legal:
            return FoldAction()
        if RaiseAction in legal:
            return RaiseAction(max(min_raise, min(max_raise, 2)))
        return CheckAction()

    # =========================================================
    # Encoding helpers
    # =========================================================

    def _encode_two(self, two_cards):
        c1, c2 = two_cards
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]

        if RANKS.index(r2) > RANKS.index(r1):
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        if r1 == r2:
            return r1 + r2

        return r1 + r2 + ("s" if s1 == s2 else "o")


if __name__ == "__main__":
    run_bot(Player(), parse_args())
