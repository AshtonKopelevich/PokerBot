'''
Hybrid FSM + Monte Carlo Poker Bot
Implements the Bot base class for the Hold'em + Redraw competition.
All decision logic lives in get_action().
'''

import random
import itertools
from collections import Counter

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot


# ---------------------------------------------------------------------------
# Globals / constants
# ---------------------------------------------------------------------------

RANKS    = '23456789TJQKA'
SUITS    = 'cdhs'
RANK_MAP = {r: i for i, r in enumerate(RANKS)}   # '2'=0 ... 'A'=12

# Preflop hand tiers — used by FSM to avoid running MC on obvious spots
TIER1 = {'AA','KK','QQ','AKs','AKo'}
TIER2 = {'JJ','TT','99','AQs','AQo','AJs','KQs','KQo'}
TIER3 = {'88','77','66','ATs','KJs','QJs','JTs','T9s','98s','AJo','KJo'}

# Monte Carlo budgets
MC_FULL  = 800   # complex postflop decisions
MC_DRAW  = 500   # per redraw candidate
MC_FAST  = 250   # quick standard postflop check

# FSM states
TRIVIAL  = 'TRIVIAL'
STANDARD = 'STANDARD'
COMPLEX  = 'COMPLEX'


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
# 5-card hand evaluator — returns a comparable tuple (higher = better)
# ---------------------------------------------------------------------------

def _eval5(cards):
    ranks   = sorted([_rank(c) for c in cards], reverse=True)
    suits   = [_suit(c) for c in cards]
    flush   = len(set(suits)) == 1
    counts  = Counter(ranks)
    freq    = sorted(counts.values(), reverse=True)
    by_freq = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
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
# Monte Carlo equity estimator
# ---------------------------------------------------------------------------

def _mc_equity(hole, board, n):
    '''Win + 0.5*tie rate over n random runouts.'''
    wins = ties = 0
    known   = hole + board
    to_deal = 5 - len(board)
    for _ in range(n):
        deck   = _fresh_deck(known)
        runout = board + deck[:to_deal]
        opp    = deck[to_deal:to_deal + 2]
        ms = _best_hand(hole + runout)
        os = _best_hand(opp  + runout)
        if   ms > os: wins += 1
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
# Bot
# ---------------------------------------------------------------------------

class Bot():
    '''
    Hybrid FSM + Monte Carlo poker bot.

    Layer 1 — FSM classifies each spot: TRIVIAL / STANDARD / COMPLEX
    Layer 2 — Monte Carlo runs only for COMPLEX spots
    Redraw   — always COMPLEX; every candidate swap is simulated
    '''

    def handle_new_round(self, game_state, round_state, active):
        self.has_redrawn = False   # one redraw per hand, reset here

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):

        # ── 1. Unpack state ──────────────────────────────────────────────
        legal     = round_state.legal_actions()
        street    = round_state.street                   # 0 pre / 3 flop / 4 turn / 5 river
        hole      = list(round_state.hands[active])      # our 2 hole cards
        board     = list(round_state.deck[:street])      # revealed community cards

        my_pip    = round_state.pips[active]
        opp_pip   = round_state.pips[1 - active]
        my_stk    = round_state.stacks[active]

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

        # Fraction of (pot + call) we must invest to continue
        pot_odds = call_cost / (pot + call_cost) if (pot + call_cost) > 0 else 0.0

        # ── 2. FSM — classify this decision ─────────────────────────────
        #
        #  TRIVIAL  → act immediately, no MC
        #  STANDARD → heuristic + optional fast MC
        #  COMPLEX  → full MC + redraw evaluation
        #
        #  Priority order:
        #    a) Preflop Tier1 or obvious trash fold  → TRIVIAL
        #    b) Redraw still available (street < 5)  → COMPLEX
        #    c) River large sizing (pot_odds > 0.30) → COMPLEX
        #    d) Close pot odds postflop (0.20–0.55)  → COMPLEX
        #    e) Everything else                      → STANDARD

        if street == 0:
            tier = _preflop_tier(hole)
            if tier == 1:
                fsm = TRIVIAL
            elif tier == 4 and pot_odds > 0.35:
                fsm = TRIVIAL
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

        # ── 3. TRIVIAL path ──────────────────────────────────────────────
        if fsm == TRIVIAL:
            if street == 0 and _preflop_tier(hole) == 1:
                amt = min(max_r, max(min_r, 3 * BIG_BLIND))
                if can_raise and amt >= min_r:
                    return RaiseAction(amt)
                return CallAction() if can_call else CheckAction()
            # Trash / all-in / no-decision
            return FoldAction() if can_fold else (CheckAction() if can_check else CallAction())

        # ── 4. STANDARD path ─────────────────────────────────────────────
        if fsm == STANDARD:
            if street == 0:
                tier = _preflop_tier(hole)
                if tier == 2:
                    amt = min(max_r, max(min_r, int(2.5 * BIG_BLIND)))
                    if can_raise and amt >= min_r and pot_odds < 0.20:
                        return RaiseAction(amt)
                    if can_call and pot_odds < 0.25: return CallAction()
                    return CheckAction() if can_check else FoldAction()
                if tier == 3:
                    if can_call and pot_odds < 0.15: return CallAction()
                    return CheckAction() if can_check else FoldAction()
                # Tier 4, cheap spot
                return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

            # Postflop standard — quick equity check
            eq = _mc_equity(hole, board, MC_FAST)

            if eq > 0.65:
                amt = min(max_r, max(min_r, int(0.6 * pot)))
                if can_raise and amt >= min_r: return RaiseAction(amt)
                return CallAction() if can_call else CheckAction()

            if eq > pot_odds + 0.05:
                if can_call and call_cost > 0: return CallAction()
                if can_check: return CheckAction()
                amt = min(max_r, max(min_r, int(0.4 * pot)))
                if can_raise and amt >= min_r: return RaiseAction(amt)

            return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        # ── 5. COMPLEX path ───────────────────────────────────────────────
        #
        # Step A  baseline equity
        # Step B  simulate every redraw candidate (if redraw still available)
        # Step C  pick base action from best equity found
        # Step D  wrap with RedrawAction if a swap gains ≥ 3% equity

        # Step A
        base_eq = _mc_equity(hole, board, MC_FULL)

        # Step B
        best_candidate = None
        best_eq        = base_eq

        if not self.has_redrawn and street < 5:

            # Try swapping each hole card
            for i in range(2):
                kept  = [hole[1 - i]]
                known = kept + board
                w = t = 0
                for _ in range(MC_DRAW):
                    deck     = _fresh_deck(known)
                    new_hole = kept + [deck[0]]
                    to_deal  = 5 - len(board)
                    runout   = board + deck[1:1 + to_deal]
                    opp      = deck[1 + to_deal:3 + to_deal]
                    ms = _best_hand(new_hole + runout)
                    os = _best_hand(opp      + runout)
                    if ms > os: w += 1
                    elif ms == os: t += 1
                eq_swap = w / MC_DRAW + 0.5 * (t / MC_DRAW)
                if eq_swap > best_eq:
                    best_eq        = eq_swap
                    best_candidate = ('hole', i)

            # Try swapping each revealed board card
            for i in range(len(board)):
                rem   = board[:i] + board[i+1:]
                known = hole + rem
                w = t = 0
                for _ in range(MC_DRAW):
                    deck      = _fresh_deck(known)
                    new_board = rem + [deck[0]]
                    to_deal   = 5 - len(new_board)
                    runout    = new_board + deck[1:1 + to_deal]
                    opp       = deck[1 + to_deal:3 + to_deal]
                    ms = _best_hand(hole + runout)
                    os = _best_hand(opp  + runout)
                    if ms > os: w += 1
                    elif ms == os: t += 1
                eq_swap = w / MC_DRAW + 0.5 * (t / MC_DRAW)
                if eq_swap > best_eq:
                    best_eq        = eq_swap
                    best_candidate = ('board', i)

        working_eq  = best_eq
        redraw_gain = best_eq - base_eq

        # Step C — equity → action
        if working_eq > 0.70:
            size = int(0.75 * pot) if street < 5 else int(1.0 * pot)
            amt  = min(max_r, max(min_r, size))
            if can_raise and amt >= min_r: base_action = RaiseAction(amt)
            elif can_call:                 base_action = CallAction()
            else:                          base_action = CheckAction()

        elif working_eq > 0.55:
            amt = min(max_r, max(min_r, int(0.45 * pot)))
            if can_raise and amt >= min_r and working_eq > pot_odds + 0.10:
                base_action = RaiseAction(amt)
            elif can_call and call_cost > 0 and working_eq > pot_odds + 0.03:
                base_action = CallAction()
            elif can_check:
                base_action = CheckAction()
            else:
                base_action = FoldAction() if can_fold else CallAction()

        elif working_eq > pot_odds + 0.02:
            base_action = CallAction() if (can_call and call_cost > 0) else (CheckAction() if can_check else FoldAction())

        else:
            base_action = CheckAction() if can_check else (FoldAction() if can_fold else CallAction())

        # Step D — attach redraw if swap gains ≥ 3%
        if best_candidate is not None and redraw_gain >= 0.03:
            self.has_redrawn = True
            ctype, idx = best_candidate
            return RedrawAction(base_action, ctype, idx)

        return base_action
