'''
Hybrid FSM + Monte Carlo + Opponent Modeling Poker Bot
=======================================================
Key fixes vs previous version:
  - Value bet proactively: always use max(min_r, ...) so we never silently
    check a strong hand just because pot-fraction math gives a tiny number
  - Lowered value bet threshold from 0.65 -> 0.55 (bet more hands for value)
  - Lowered strong call threshold from 0.55 -> 0.50 (call wider postflop)
  - Added turn/river bluffing (not just flop)
  - Preflop: TIER2 now always raises, TIER3 calls rather than folding
  - Redraw logic preserved and guarded with RedrawAction in legal check
'''

import random
import itertools
from collections import Counter

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANKS    = '23456789TJQKA'
SUITS    = 'cdhs'
RANK_MAP = {r: i for i, r in enumerate(RANKS)}

TIER1 = {'AA','KK','QQ','AKs','AKo'}
TIER2 = {'JJ','TT','99','AQs','AQo','AJs','KQs','KQo'}
TIER3 = {'88','77','66','ATs','KJs','QJs','JTs','T9s','98s','AJo','KJo'}

MC_FULL  = 800
MC_DRAW  = 500
MC_FAST  = 250

TRIVIAL  = 'TRIVIAL'
STANDARD = 'STANDARD'
COMPLEX  = 'COMPLEX'

MODEL_MIN_HANDS = 15


# ---------------------------------------------------------------------------
# Card / deck helpers
# ---------------------------------------------------------------------------

def _fresh_deck(exclude):
    ex   = set(exclude)
    deck = [r + s for r in RANKS for s in SUITS if (r + s) not in ex]
    random.shuffle(deck)
    return deck

def _rank(card): return RANK_MAP[card[0]]
def _suit(card): return card[1]


# ---------------------------------------------------------------------------
# Hand evaluator
# ---------------------------------------------------------------------------

def _eval5(cards):
    ranks    = sorted([_rank(c) for c in cards], reverse=True)
    suits    = [_suit(c) for c in cards]
    flush    = len(set(suits)) == 1
    counts   = Counter(ranks)
    freq     = sorted(counts.values(), reverse=True)
    by_freq  = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    straight = len(set(ranks)) == 5 and ranks[0] - ranks[4] == 4
    wheel    = set(ranks) == {12, 0, 1, 2, 3}

    if flush and (straight or wheel):  return (8, 3 if wheel else ranks[0])
    if freq[0] == 4:                   return (7, by_freq[0], by_freq[1])
    if freq[:2] == [3, 2]:             return (6, by_freq[0], by_freq[1])
    if flush:                          return (5,) + tuple(ranks)
    if straight:                       return (4, ranks[0])
    if wheel:                          return (4, 3)
    if freq[0] == 3:                   return (3, by_freq[0]) + tuple(by_freq[1:3])
    if freq[:2] == [2, 2]:
        p1, p2 = sorted(by_freq[:2], reverse=True)
        return (2, p1, p2, by_freq[2])
    if freq[0] == 2:                   return (1, by_freq[0]) + tuple(by_freq[1:4])
    return (0,) + tuple(ranks)

def _best_hand(cards):
    return max(_eval5(c) for c in itertools.combinations(cards, 5))


# ---------------------------------------------------------------------------
# Monte Carlo equity
# ---------------------------------------------------------------------------

def _mc_equity(hole, board, n):
    wins = ties = 0
    known   = hole + board
    to_deal = 5 - len(board)
    for _ in range(n):
        deck   = _fresh_deck(known)
        runout = board + deck[:to_deal]
        opp    = deck[to_deal:to_deal + 2]
        ms = _best_hand(hole + runout)
        os = _best_hand(opp  + runout)
        if   ms > os:  wins += 1
        elif ms == os: ties += 1
    return wins / n + 0.5 * (ties / n)


# ---------------------------------------------------------------------------
# Preflop hand key + tier
# ---------------------------------------------------------------------------

def _preflop_key(hole):
    r0, r1 = _rank(hole[0]), _rank(hole[1])
    s0, s1 = _suit(hole[0]), _suit(hole[1])
    hi, lo = (hole[0], hole[1]) if r0 >= r1 else (hole[1], hole[0])
    if r0 == r1:
        return hi[0] + lo[0]
    return hi[0] + lo[0] + ('s' if s0 == s1 else 'o')

def _preflop_tier(hole):
    key = _preflop_key(hole)
    if key in TIER1: return 1
    if key in TIER2: return 2
    if key in TIER3: return 3
    return 4


# ---------------------------------------------------------------------------
# Helper: make a value bet amount, always at least min_r
# FIX: old code did min(max_r, size) which could produce 0 when pot is tiny,
# causing us to silently check strong hands. Now we clamp UP to min_r too.
# ---------------------------------------------------------------------------

def _bet_amount(pot, fraction, multiplier, min_r, max_r):
    size = int(pot * fraction * multiplier)
    return min(max_r, max(min_r, size))


# ---------------------------------------------------------------------------
# Opponent model
# ---------------------------------------------------------------------------

class OpponentModel:
    def __init__(self):
        self.hands_played     = 0
        self.vpip_events      = 0
        self.vpip_opps        = 0
        self.fold_3b_events   = 0
        self.fold_3b_opps     = 0
        self.cbet_fold_events = 0
        self.cbet_fold_opps   = 0
        self.agg_events       = 0
        self.agg_opps         = 0
        self.wtsd_events      = 0
        self.wtsd_opps        = 0
        self.redraw_events    = 0
        self.redraw_opps      = 0
        self.showdown_hands   = []
        self.this_hand        = {}

    def _rate(self, events, opps, default=0.5):
        if opps < 5:
            return default
        return events / opps

    @property
    def vpip(self):         return self._rate(self.vpip_events,      self.vpip_opps,      0.5)
    @property
    def fold_to_3bet(self): return self._rate(self.fold_3b_events,   self.fold_3b_opps,   0.5)
    @property
    def fold_to_cbet(self): return self._rate(self.cbet_fold_events, self.cbet_fold_opps, 0.5)
    @property
    def aggression(self):   return self._rate(self.agg_events,       self.agg_opps,       0.5)
    @property
    def wtsd(self):         return self._rate(self.wtsd_events,      self.wtsd_opps,      0.5)
    @property
    def redraw_rate(self):  return self._rate(self.redraw_events,    self.redraw_opps,    0.5)
    @property
    def is_reliable(self):  return self.hands_played >= MODEL_MIN_HANDS

    def equity_call_adjustment(self):
        if not self.is_reliable: return 0.0
        return (self.wtsd - 0.50) * 0.20

    def bluff_size_multiplier(self):
        if not self.is_reliable: return 1.0
        return 0.6 + self.fold_to_cbet * 1.0

    def value_size_multiplier(self):
        if not self.is_reliable: return 1.0
        return 0.7 + self.wtsd * 0.6

    def preflop_raise_adjustment(self):
        if not self.is_reliable: return 0
        return int((self.vpip - 0.5) * 2 * BIG_BLIND)

    def redraw_threshold_adjustment(self):
        if not self.is_reliable: return 0.0
        return (0.50 - self.aggression) * 0.04

    def record_preflop_action(self, opp_voluntarily_played):
        self.vpip_opps += 1
        if opp_voluntarily_played:
            self.vpip_events += 1
        self.redraw_opps += 1

    def record_fold_to_raise(self, folded):
        self.fold_3b_opps += 1
        if folded: self.fold_3b_events += 1

    def record_postflop_action(self, opp_was_aggressive):
        self.agg_opps += 1
        if opp_was_aggressive: self.agg_events += 1

    def record_cbet_response(self, opp_folded):
        self.cbet_fold_opps += 1
        if opp_folded: self.cbet_fold_events += 1

    def record_redraw(self, used_redraw):
        if used_redraw: self.redraw_events += 1

    def record_showdown(self, opp_hole_cards, went_to_showdown):
        self.wtsd_opps += 1
        if went_to_showdown:
            self.wtsd_events += 1
            if opp_hole_cards:
                self.showdown_hands.append(_preflop_key(opp_hole_cards))

    def end_hand(self):
        self.hands_played += 1
        self.this_hand = {}


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class Bot(Bot):
    def __init__(self):
        self.opp                = OpponentModel()
        self.has_redrawn        = False
        self._we_raised_preflop = False
        self._we_cbet_flop      = False
        self._saw_flop          = False
        self._opp_redrawn       = False

    def handle_new_round(self, game_state, round_state, active):
        self.has_redrawn        = False
        self._we_raised_preflop = False
        self._we_cbet_flop      = False
        self._saw_flop          = False
        self._opp_redrawn       = False

    def handle_round_over(self, game_state, terminal_state, active):
        opp_idx    = 1 - active
        prev_state = terminal_state.previous_state
        deltas     = terminal_state.deltas

        opp_pip_preflop    = prev_state.pips[opp_idx] if prev_state.street == 0 else BIG_BLIND
        opp_played_preflop = opp_pip_preflop > BIG_BLIND
        self.opp.record_preflop_action(opp_played_preflop)

        if self._we_raised_preflop:
            opp_folded_preflop = (
                deltas[active] > 0 and
                prev_state.street == 0 and
                prev_state.pips[opp_idx] < prev_state.pips[active]
            )
            self.opp.record_fold_to_raise(opp_folded_preflop)

        if self._we_cbet_flop and self._saw_flop:
            opp_folded_flop = (deltas[active] > 0 and prev_state.street <= 3)
            self.opp.record_cbet_response(opp_folded_flop)

        if self._saw_flop:
            opp_was_aggressive = prev_state.pips[opp_idx] > prev_state.pips[active]
            self.opp.record_postflop_action(opp_was_aggressive)

        self.opp.record_redraw(self._opp_redrawn)

        opp_hole         = None
        went_to_showdown = False
        if (
            hasattr(prev_state, 'hands') and
            prev_state.hands is not None and
            len(prev_state.hands) > opp_idx and
            prev_state.hands[opp_idx] and
            len(prev_state.hands[opp_idx]) == 2 and
            prev_state.street == 5
        ):
            opp_hole         = list(prev_state.hands[opp_idx])
            went_to_showdown = True

        self.opp.record_showdown(opp_hole if went_to_showdown else None, went_to_showdown)
        self.opp.end_hand()

    def get_action(self, game_state, round_state, active):

        # ── 1. Unpack state ──────────────────────────────────────────────────
        legal     = round_state.legal_actions()
        street    = round_state.street
        hole      = list(round_state.hands[active])
        board     = list(round_state.deck[:street])

        my_pip    = round_state.pips[active]
        opp_pip   = round_state.pips[1 - active]
        call_cost = max(0, opp_pip - my_pip)
        pot       = my_pip + opp_pip

        if RaiseAction in legal:
            min_r, max_r = round_state.raise_bounds()
        else:
            min_r = max_r = 0

        can_raise = RaiseAction in legal
        can_call  = CallAction  in legal
        can_check = CheckAction in legal
        can_fold  = FoldAction  in legal

        denom    = pot + call_cost
        pot_odds = call_cost / denom if denom > 0 else 0.0

        if street >= 3:
            self._saw_flop = True

        # ── 2. Opponent model adjustments ───────────────────────────────────
        eq_adj       = self.opp.equity_call_adjustment()
        bluff_mult   = self.opp.bluff_size_multiplier()
        value_mult   = self.opp.value_size_multiplier()
        pf_raise_adj = self.opp.preflop_raise_adjustment()
        redraw_adj   = self.opp.redraw_threshold_adjustment()

        # FIX: lowered thresholds vs old version
        # Old value_threshold = 0.65 → now 0.55  (bet more hands for value)
        # Old call_threshold  = 0.55 → now 0.50  (call wider)
        value_threshold       = 0.55
        call_threshold_strong = 0.50 + eq_adj
        redraw_threshold      = max(0.01, 0.03 - redraw_adj)

        # ── 3. FSM ───────────────────────────────────────────────────────────
        if street == 0:
            tier = _preflop_tier(hole)
            if tier <= 2:
                fsm = TRIVIAL          # FIX: TIER2 promoted to TRIVIAL (always raise)
            elif tier == 4 and pot_odds > 0.35:
                fsm = TRIVIAL          # garbage vs large bet → fold
            else:
                fsm = STANDARD
        elif not self.has_redrawn and street < 5:
            fsm = COMPLEX
        elif street == 5 and pot_odds > 0.30:
            fsm = COMPLEX
        elif 0.20 < pot_odds < 0.55:
            fsm = COMPLEX
        else:
            fsm = STANDARD

        # ── 4. TRIVIAL path ──────────────────────────────────────────────────
        if fsm == TRIVIAL:
            if street == 0:
                tier = _preflop_tier(hole)
                if tier <= 2:
                    # FIX: TIER2 raises too (was only TIER1 before)
                    base = (3 * BIG_BLIND if tier == 1 else int(2.5 * BIG_BLIND)) + pf_raise_adj
                    amt  = min(max_r, max(min_r, base))
                    if can_raise and max_r >= min_r:
                        self._we_raised_preflop = True
                        return RaiseAction(amt)
                    return CallAction() if can_call else CheckAction()
            # Garbage hand vs large bet
            return FoldAction() if can_fold else (CheckAction() if can_check else CallAction())

        # ── 5. STANDARD path ─────────────────────────────────────────────────
        if fsm == STANDARD:
            if street == 0:
                tier = _preflop_tier(hole)
                if tier == 3:
                    # FIX: TIER3 calls small bets rather than folding everything
                    if can_call and pot_odds < 0.20:
                        return CallAction()
                    return CheckAction() if can_check else FoldAction()
                # TIER4 — fold to bets, check free
                return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

            # Postflop standard — quick equity check
            eq = _mc_equity(hole, board, MC_FAST)

            if eq >= value_threshold:
                # FIX: _bet_amount clamps UP to min_r — we always bet strong hands
                fraction = 0.75 if eq > 0.70 else 0.5
                if can_raise and max_r >= min_r:
                    amt = _bet_amount(pot, fraction, value_mult, min_r, max_r)
                    if street == 3:
                        self._we_cbet_flop = True
                    return RaiseAction(amt)
                return CallAction() if can_call else CheckAction()

            if eq >= call_threshold_strong:
                if can_call and call_cost > 0:
                    return CallAction()
                # Medium hand, nobody bet — small probe
                if can_raise and max_r >= min_r:
                    amt = _bet_amount(pot, 0.35, bluff_mult, min_r, max_r)
                    if street == 3:
                        self._we_cbet_flop = True
                    return RaiseAction(amt)
                return CheckAction() if can_check else FoldAction()

            return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        # ── 6. COMPLEX path ───────────────────────────────────────────────────

        # Step A — baseline equity
        base_eq = _mc_equity(hole, board, MC_FULL)

        # Step B — redraw search
        best_candidate = None
        best_eq        = base_eq

        if not self.has_redrawn and street < 5 and RedrawAction in legal:

            for i in range(2):
                kept  = [hole[1 - i]]
                known = kept + board
                w = t = 0
                for _ in range(MC_DRAW):
                    deck     = _fresh_deck(known)
                    new_hole = kept + [deck[0]]
                    to_deal  = 5 - len(board)
                    runout   = board + deck[1:1 + to_deal]
                    opp_hand = deck[1 + to_deal:3 + to_deal]
                    ms = _best_hand(new_hole + runout)
                    os = _best_hand(opp_hand + runout)
                    if ms > os:  w += 1
                    elif ms == os: t += 1
                eq_swap = w / MC_DRAW + 0.5 * (t / MC_DRAW)
                if eq_swap > best_eq:
                    best_eq        = eq_swap
                    best_candidate = ('hole', i)

            for i in range(len(board)):
                rem   = board[:i] + board[i+1:]
                known = hole + rem
                w = t = 0
                for _ in range(MC_DRAW):
                    deck      = _fresh_deck(known)
                    new_board = rem + [deck[0]]
                    to_deal   = 5 - len(new_board)
                    runout    = new_board + deck[1:1 + to_deal]
                    opp_hand  = deck[1 + to_deal:3 + to_deal]
                    ms = _best_hand(hole + runout)
                    os = _best_hand(opp_hand + runout)
                    if ms > os:  w += 1
                    elif ms == os: t += 1
                eq_swap = w / MC_DRAW + 0.5 * (t / MC_DRAW)
                if eq_swap > best_eq:
                    best_eq        = eq_swap
                    best_candidate = ('board', i)

        working_eq  = best_eq
        redraw_gain = best_eq - base_eq

        # Step C — equity → base action
        if working_eq >= value_threshold and can_raise and max_r >= min_r:
            # FIX: _bet_amount clamps UP to min_r — never silently check a strong hand
            fraction    = (1.0 if street == 5 else 0.75) if working_eq > 0.70 else 0.5
            amt         = _bet_amount(pot, fraction, value_mult, min_r, max_r)
            if street == 3:
                self._we_cbet_flop = True
            base_action = RaiseAction(amt)

        elif working_eq >= value_threshold:
            base_action = CallAction() if can_call else CheckAction()

        elif working_eq >= call_threshold_strong:
            amt = _bet_amount(pot, 0.45, value_mult, min_r, max_r)
            if can_raise and max_r >= min_r and working_eq > pot_odds + 0.08:
                if street == 3:
                    self._we_cbet_flop = True
                base_action = RaiseAction(amt)
            elif can_call and call_cost > 0 and working_eq > pot_odds + eq_adj:
                base_action = CallAction()
            elif can_check:
                base_action = CheckAction()
            else:
                base_action = FoldAction() if can_fold else CallAction()

        elif working_eq > pot_odds + eq_adj + 0.02:
            if can_call and call_cost > 0:
                base_action = CallAction()
            else:
                base_action = CheckAction() if can_check else FoldAction()

        elif (working_eq < 0.40 and
              self.opp.fold_to_cbet > 0.55 and
              street in (3, 4) and          # FIX: bluff on turn too, not just flop
              can_raise and max_r >= min_r):
            amt = _bet_amount(pot, 0.5, bluff_mult, min_r, max_r)
            if street == 3:
                self._we_cbet_flop = True
            base_action = RaiseAction(amt)

        else:
            base_action = CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        # Step D — attach redraw if gain clears the threshold
        if best_candidate is not None and redraw_gain >= redraw_threshold and RedrawAction in legal:
            self.has_redrawn = True
            ctype, idx = best_candidate
            return RedrawAction(base_action, ctype, idx)

        return base_action
