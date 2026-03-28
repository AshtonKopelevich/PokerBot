'''
Adaptive Poker Bot — FSM + Monte Carlo + Opponent Style Detection
=================================================================
Detects opponent playstyle from observed actions and switches strategy:

  STYLE_UNKNOWN   — first ~10 hands, use neutral defaults
  STYLE_AGGRESSIVE — all-in / large raise bot
      → call wide, never fold to big bets with decent equity
      → value bet big (they call everything)
      → redraw aggressively to improve before calling off stack

  STYLE_PASSIVE   — check/call bot (folds to raises > ~8 chips)
      → value bet SMALL (≤ 8 chips) so they call instead of fold
      → never bluff (they only call when ahead)
      → check back mediocre hands, get to showdown cheap
      → raise threshold: only raise when equity > 0.65

  STYLE_BALANCED  — unknown / mixed, use standard MC thresholds

Detection signals (updated every hand):
  - opp_raise_freq   : how often they raise postflop
  - opp_fold_freq    : how often they fold when we bet
  - opp_avg_bet_size : average bet size relative to pot
  - opp_vpip         : voluntary preflop investment rate
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

MC_FULL = 600
MC_DRAW = 400
MC_FAST = 200

STYLE_UNKNOWN    = 'UNKNOWN'
STYLE_AGGRESSIVE = 'AGGRESSIVE'
STYLE_PASSIVE    = 'PASSIVE'
STYLE_BALANCED   = 'BALANCED'

# Chips — against passive bot they fold raises above this
PASSIVE_MAX_BET = 8


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
# Preflop helpers
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
# Bet sizing helper — always clamps UP to min_r
# ---------------------------------------------------------------------------

def _bet(pot, fraction, multiplier, min_r, max_r):
    size = int(pot * fraction * multiplier)
    return min(max_r, max(min_r, size))


# ---------------------------------------------------------------------------
# Opponent style tracker
# ---------------------------------------------------------------------------

class StyleTracker:
    '''
    Watches what the opponent does over every hand and classifies them
    into AGGRESSIVE, PASSIVE, or BALANCED after enough data.

    Key signals:
      raise_opps / raise_events  → do they ever raise?
      fold_opps  / fold_events   → do they fold when we bet?
      bet_sizes                  → how big are their bets relative to pot?
      vpip_events / vpip_opps    → do they invest preflop voluntarily?
    '''

    def __init__(self):
        self.hands = 0

        self.raise_events = 0   # times opp raised postflop
        self.raise_opps   = 0   # postflop action opportunities
        self.fold_events  = 0   # times opp folded when we bet
        self.fold_opps    = 0   # times we bet and they had a decision
        self.vpip_events  = 0
        self.vpip_opps    = 0
        self.bet_sizes    = []  # relative bet sizes seen (opp_pip / pot)

        # Per-hand scratch
        self._we_bet_this_street = False

    def _rate(self, ev, op, default=0.5):
        return ev / op if op >= 5 else default

    @property
    def raise_freq(self):
        return self._rate(self.raise_events, self.raise_opps, default=0.5)

    @property
    def fold_freq(self):
        return self._rate(self.fold_events, self.fold_opps, default=0.5)

    @property
    def vpip(self):
        return self._rate(self.vpip_events, self.vpip_opps, default=0.5)

    @property
    def avg_bet_size(self):
        if len(self.bet_sizes) < 5:
            return 0.5  # neutral
        return sum(self.bet_sizes) / len(self.bet_sizes)

    @property
    def style(self):
        if self.hands < 10:
            return STYLE_UNKNOWN

        # Aggressive: raises a lot, bets big, doesn't fold
        if self.raise_freq > 0.35 or self.avg_bet_size > 1.5:
            return STYLE_AGGRESSIVE

        # Passive: almost never raises, folds to our bets often, low vpip
        if self.raise_freq < 0.10 and self.fold_freq > 0.40:
            return STYLE_PASSIVE

        # Also passive if they almost never voluntarily put chips in
        if self.vpip < 0.25 and self.raise_freq < 0.15:
            return STYLE_PASSIVE

        return STYLE_BALANCED

    def record_opp_action(self, opp_raised, opp_folded, we_bet,
                          opp_bet_fraction=None, opp_vpip=None):
        self.raise_opps += 1
        if opp_raised:
            self.raise_events += 1
        if we_bet:
            self.fold_opps += 1
            if opp_folded:
                self.fold_events += 1
        if opp_bet_fraction is not None:
            self.bet_sizes.append(opp_bet_fraction)
        if opp_vpip is not None:
            self.vpip_opps += 1
            if opp_vpip:
                self.vpip_events += 1

    def end_hand(self):
        self.hands += 1
        self._we_bet_this_street = False


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class Bot(Bot):
    def __init__(self):
        self.tracker        = StyleTracker()
        self.has_redrawn    = False

        # Per-hand scratch for style tracker
        self._we_bet_postflop   = False
        self._saw_flop          = False
        self._opp_raised_hand   = False
        self._opp_pip_at_start  = 0
        self._pot_at_bet        = 0
        self._we_raised_preflop = False

    def handle_new_round(self, game_state, round_state, active):
        self.has_redrawn        = False
        self._we_bet_postflop   = False
        self._saw_flop          = False
        self._opp_raised_hand   = False
        self._opp_pip_at_start  = 0
        self._pot_at_bet        = 0
        self._we_raised_preflop = False

    def handle_round_over(self, game_state, terminal_state, active):
        opp_idx    = 1 - active
        prev       = terminal_state.previous_state
        deltas     = terminal_state.deltas

        # Did opponent voluntarily invest preflop?
        opp_pip_pf = prev.pips[opp_idx] if prev.street == 0 else BIG_BLIND
        opp_vpip   = opp_pip_pf > BIG_BLIND

        # Did opponent raise postflop?
        opp_raised = self._opp_raised_hand

        # Did opponent fold to our bet?
        opp_folded = (
            self._we_bet_postflop and
            deltas[active] > 0 and
            prev.street in (3, 4, 5)
        )

        # Opponent bet size relative to pot (rough)
        opp_bet_frac = None
        if self._saw_flop and self._pot_at_bet > 0:
            opp_contrib = prev.pips[opp_idx]
            if opp_contrib > BIG_BLIND:
                opp_bet_frac = min(4.0, opp_contrib / max(1, self._pot_at_bet))

        self.tracker.record_opp_action(
            opp_raised    = opp_raised,
            opp_folded    = opp_folded,
            we_bet        = self._we_bet_postflop,
            opp_bet_fraction = opp_bet_frac,
            opp_vpip      = opp_vpip,
        )
        self.tracker.end_hand()

    def get_action(self, game_state, round_state, active):

        # ── 1. State ─────────────────────────────────────────────────────────
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
            if self._pot_at_bet == 0:
                self._pot_at_bet = pot

        # Track if opp raised this hand
        if opp_pip > my_pip + BIG_BLIND and street > 0:
            self._opp_raised_hand = True

        # ── 2. Detect opponent style ─────────────────────────────────────────
        style = self.tracker.style

        # ── 3. Set thresholds based on style ─────────────────────────────────
        if style == STYLE_AGGRESSIVE:
            # They go all-in constantly — just need 50%+ equity to call/raise
            value_threshold  = 0.52   # bet/raise for value
            call_threshold   = 0.48   # call threshold (they bluff a ton)
            bluff_ok         = False  # no point bluffing a caller
            redraw_threshold = 0.02   # redraw aggressively before calling off

        elif style == STYLE_PASSIVE:
            # They fold to bets > 8 chips — bet small with strong hands, never bluff
            value_threshold  = 0.60   # only bet when clearly ahead
            call_threshold   = 0.50   # call their tiny bets wide
            bluff_ok         = False  # NEVER bluff — they only call when ahead
            redraw_threshold = 0.03

        else:
            # Unknown / balanced — moderate thresholds
            value_threshold  = 0.55
            call_threshold   = 0.50
            bluff_ok         = True
            redraw_threshold = 0.03

        # ── 4. Preflop ───────────────────────────────────────────────────────
        if street == 0:
            tier = _preflop_tier(hole)

            if style == STYLE_AGGRESSIVE:
                # Call any raise with top hands, fold junk — they're going all-in anyway
                if tier <= 2:
                    amt = min(max_r, max(min_r, 3 * BIG_BLIND))
                    if can_raise and max_r >= min_r:
                        self._we_raised_preflop = True
                        return RaiseAction(amt)
                    return CallAction() if can_call else CheckAction()
                if tier == 3:
                    return CallAction() if can_call else CheckAction()
                # Tier 4 — fold to large bets, check free
                if pot_odds > 0.30:
                    return FoldAction() if can_fold else CheckAction()
                return CheckAction() if can_check else CallAction()

            elif style == STYLE_PASSIVE:
                # They fold to big raises, small raises or call with playable hands
                if tier <= 2:
                    # Small raise — they'll fold to big ones anyway
                    amt = min(max_r, max(min_r, int(2 * BIG_BLIND)))
                    if can_raise and max_r >= min_r:
                        self._we_raised_preflop = True
                        return RaiseAction(amt)
                    return CallAction() if can_call else CheckAction()
                if tier == 3:
                    return CallAction() if can_call else CheckAction()
                return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

            else:
                # Balanced / unknown
                if tier <= 2:
                    amt = min(max_r, max(min_r, int(2.5 * BIG_BLIND)))
                    if can_raise and max_r >= min_r:
                        self._we_raised_preflop = True
                        return RaiseAction(amt)
                    return CallAction() if can_call else CheckAction()
                if tier == 3:
                    if can_call and pot_odds < 0.20:
                        return CallAction()
                    return CheckAction() if can_check else FoldAction()
                return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        # ── 5. Postflop — run equity ─────────────────────────────────────────
        n_sims   = MC_FULL if (not self.has_redrawn and street < 5) else MC_FAST
        base_eq  = _mc_equity(hole, board, n_sims)

        # ── 6. Redraw search ─────────────────────────────────────────────────
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
                    opp_h    = deck[1 + to_deal:3 + to_deal]
                    ms = _best_hand(new_hole + runout)
                    os = _best_hand(opp_h    + runout)
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
                    opp_h     = deck[1 + to_deal:3 + to_deal]
                    ms = _best_hand(hole  + runout)
                    os = _best_hand(opp_h + runout)
                    if ms > os:  w += 1
                    elif ms == os: t += 1
                eq_swap = w / MC_DRAW + 0.5 * (t / MC_DRAW)
                if eq_swap > best_eq:
                    best_eq        = eq_swap
                    best_candidate = ('board', i)

        eq          = best_eq
        redraw_gain = best_eq - base_eq

        # ── 7. Postflop action — style-specific ──────────────────────────────

        if style == STYLE_AGGRESSIVE:
            # They go all-in — call wide, raise for value with strong hands
            if eq >= value_threshold:
                if can_raise and max_r >= min_r:
                    # Bet big — they'll call with anything
                    amt = _bet(pot, 0.75, 1.0, min_r, max_r)
                    self._we_bet_postflop = True
                    base_action = RaiseAction(amt)
                elif can_call:
                    base_action = CallAction()
                else:
                    base_action = CheckAction()

            elif eq >= call_threshold:
                # Decent hand — call their shoves, check if no action
                if can_call and call_cost > 0:
                    base_action = CallAction()
                elif can_raise and max_r >= min_r:
                    amt = _bet(pot, 0.5, 1.0, min_r, max_r)
                    self._we_bet_postflop = True
                    base_action = RaiseAction(amt)
                else:
                    base_action = CheckAction()

            elif eq >= pot_odds + 0.03:
                # Thin call — pot odds justify it
                base_action = CallAction() if (can_call and call_cost > 0) else (CheckAction() if can_check else FoldAction())

            else:
                base_action = CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        elif style == STYLE_PASSIVE:
            # They fold to raises > 8 chips — must bet small to get paid
            if eq >= value_threshold:
                if can_raise and max_r >= min_r:
                    # KEY FIX: cap bet at PASSIVE_MAX_BET so they actually call
                    amt = min(PASSIVE_MAX_BET, min(max_r, max(min_r, int(0.4 * pot))))
                    amt = max(min_r, amt)  # still must be legal
                    self._we_bet_postflop = True
                    base_action = RaiseAction(amt)
                elif can_call:
                    base_action = CallAction()
                else:
                    base_action = CheckAction()

            elif eq >= call_threshold:
                # Medium hand — check/call, never bet (they'll fold or have us beat)
                if can_call and call_cost > 0 and call_cost <= PASSIVE_MAX_BET:
                    base_action = CallAction()
                else:
                    base_action = CheckAction() if can_check else FoldAction()

            elif eq >= pot_odds + 0.02:
                base_action = CallAction() if (can_call and call_cost > 0 and call_cost <= PASSIVE_MAX_BET) else (CheckAction() if can_check else FoldAction())

            else:
                # Weak hand — never bluff passive bot, just check/fold
                base_action = CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        else:
            # Balanced / unknown — standard MC-based play
            if eq >= value_threshold and can_raise and max_r >= min_r:
                fraction = 0.75 if eq > 0.70 else 0.5
                amt      = _bet(pot, fraction, 1.0, min_r, max_r)
                self._we_bet_postflop = True
                base_action = RaiseAction(amt)

            elif eq >= value_threshold:
                base_action = CallAction() if can_call else CheckAction()

            elif eq >= call_threshold:
                if can_call and call_cost > 0:
                    base_action = CallAction()
                elif can_raise and max_r >= min_r:
                    amt = _bet(pot, 0.35, 1.0, min_r, max_r)
                    self._we_bet_postflop = True
                    base_action = RaiseAction(amt)
                else:
                    base_action = CheckAction() if can_check else FoldAction()

            elif eq >= pot_odds + 0.02:
                base_action = CallAction() if (can_call and call_cost > 0) else (CheckAction() if can_check else FoldAction())

            elif bluff_ok and eq < 0.38 and street in (3, 4) and can_raise and max_r >= min_r:
                amt = _bet(pot, 0.5, 1.0, min_r, max_r)
                self._we_bet_postflop = True
                base_action = RaiseAction(amt)

            else:
                base_action = CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        # ── 8. Attach redraw if beneficial ───────────────────────────────────
        if best_candidate is not None and redraw_gain >= redraw_threshold and RedrawAction in legal:
            self.has_redrawn = True
            ctype, idx = best_candidate
            return RedrawAction(base_action, ctype, idx)

        return base_action
