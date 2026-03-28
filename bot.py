'''
Hybrid FSM + Monte Carlo + Opponent Modeling Poker Bot
=======================================================
Architecture:
  Layer 1  — FSM classifies each spot: TRIVIAL / STANDARD / COMPLEX
  Layer 2  — Monte Carlo equity simulation (only for COMPLEX spots)
  Layer 3  — Opponent model built from handle_round_over(), adjusts
             bluff frequency, call thresholds, and bet sizing live
 
Opponent stats tracked (updated every hand in handle_round_over):
  fold_to_raise     — how often they fold when we raise preflop
  fold_to_cbet      — how often they fold to our flop bet
  aggression_freq   — how often they bet/raise vs check/call postflop
  vpip              — how often they voluntarily put chips in preflop
  showdown_hands    — actual hole cards seen at showdown (for range reads)
  redraw_count      — how often they use their redraw
  wtsd              — went to showdown frequency (calling station indicator)
 
All stats feed multipliers applied to:
  - equity thresholds (call more vs folders, less vs stations)
  - bluff bet sizing (bluff more vs tight players)
  - redraw threshold (lower bar vs aggressive opponents)
'''
 
import random
import itertools
from collections import Counter
 
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
 
 
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
 
# How many hands before we trust the opponent model enough to use it
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
# Opponent model — all stats live here
# ---------------------------------------------------------------------------
 
class OpponentModel:
    '''
    Tracks opponent tendencies across all 300 hands.
    Updated in handle_round_over(), queried in get_action().
 
    Every stat is stored as (events, opportunities) so we always
    know how reliable the estimate is. We fall back to neutral
    defaults until MODEL_MIN_HANDS hands have been played.
    '''
 
    def __init__(self):
        self.hands_played = 0
 
        # Preflop
        self.vpip_events   = 0   # voluntarily put chips in preflop
        self.vpip_opps     = 0
 
        self.fold_3b_events = 0  # folded when we raised preflop
        self.fold_3b_opps   = 0
 
        # Postflop
        self.cbet_fold_events = 0   # folded to our flop bet
        self.cbet_fold_opps   = 0
 
        self.agg_events = 0   # bet or raised (postflop)
        self.agg_opps   = 0   # any postflop action opportunity
 
        # Showdown / general
        self.wtsd_events = 0  # went to showdown
        self.wtsd_opps   = 0  # saw the flop (opportunity to go to showdown)
 
        self.redraw_events = 0  # used their redraw
        self.redraw_opps   = 0
 
        # Actual hands seen at showdown — for range reads
        self.showdown_hands = []   # list of hand-key strings e.g. ['AKs', '72o']
 
        # Per-hand scratch (filled during the hand, committed at round_over)
        self.this_hand = {}
 
    # ── Stat accessors (return float in [0,1], default to 0.5 if too few samples) ──
 
    def _rate(self, events, opps, default=0.5):
        if opps < 5:
            return default
        return events / opps
 
    @property
    def vpip(self):
        '''How often they voluntarily play preflop. High = loose, low = tight.'''
        return self._rate(self.vpip_events, self.vpip_opps, default=0.5)
 
    @property
    def fold_to_3bet(self):
        '''How often they fold when we raise preflop.'''
        return self._rate(self.fold_3b_events, self.fold_3b_opps, default=0.5)
 
    @property
    def fold_to_cbet(self):
        '''How often they fold to our continuation bet on the flop.'''
        return self._rate(self.cbet_fold_events, self.cbet_fold_opps, default=0.5)
 
    @property
    def aggression(self):
        '''Postflop bet/raise frequency. High = aggressive, low = passive.'''
        return self._rate(self.agg_events, self.agg_opps, default=0.5)
 
    @property
    def wtsd(self):
        '''Went-to-showdown rate. High = calling station, low = folder.'''
        return self._rate(self.wtsd_events, self.wtsd_opps, default=0.5)
 
    @property
    def redraw_rate(self):
        '''How often they use their redraw.'''
        return self._rate(self.redraw_events, self.redraw_opps, default=0.5)
 
    @property
    def is_reliable(self):
        return self.hands_played >= MODEL_MIN_HANDS
 
    # ── Derived strategy adjustments ─────────────────────────────────────────
 
    def equity_call_adjustment(self):
        '''
        Adjust the equity threshold we need to call.
        vs a calling station (high wtsd): tighten up, only call with real equity
        vs a folder (low wtsd):           loosen up, they often fold so we need less
        Returns a float added to our equity threshold. Negative = looser call.
        '''
        if not self.is_reliable:
            return 0.0
        # wtsd=0.7 (station) → +0.05 (need more equity to call)
        # wtsd=0.3 (folder)  → -0.05 (need less equity to call)
        return (self.wtsd - 0.50) * 0.20
 
    def bluff_size_multiplier(self):
        '''
        Scale our bluff bet sizing based on how often they fold.
        vs tight (high fold_to_cbet): bluff bigger, they fold anyway
        vs station (low fold_to_cbet): bluff smaller or not at all
        Returns a multiplier on pot fraction. 1.0 = no adjustment.
        '''
        if not self.is_reliable:
            return 1.0
        # fold_to_cbet=0.7 → 1.3 (bet bigger)
        # fold_to_cbet=0.3 → 0.7 (bet smaller)
        return 0.6 + self.fold_to_cbet * 1.0
 
    def value_size_multiplier(self):
        '''
        Scale our value bet sizing based on how much they call.
        vs station (high wtsd): bet bigger for value, they'll call
        vs folder (low wtsd):   bet smaller so they don't fold
        '''
        if not self.is_reliable:
            return 1.0
        return 0.7 + self.wtsd * 0.6
 
    def preflop_raise_adjustment(self):
        '''
        Adjust preflop raise sizing.
        vs loose (high vpip): raise bigger to punish wide range
        vs tight (low vpip):  raise smaller, they only continue strong
        Returns chips to add to base raise size.
        '''
        if not self.is_reliable:
            return 0
        return int((self.vpip - 0.5) * 2 * BIG_BLIND)
 
    def redraw_threshold_adjustment(self):
        '''
        Lower the redraw equity gain threshold vs aggressive opponents
        because improving our hand matters more against someone who keeps
        betting into us.
        Returns a float subtracted from the 0.03 default threshold.
        '''
        if not self.is_reliable:
            return 0.0
        # aggression=0.7 → -0.01 (redraw more readily)
        # aggression=0.3 → +0.01 (be more selective)
        return (0.50 - self.aggression) * 0.04
 
    # ── Per-hand tracking helpers ─────────────────────────────────────────────
 
    def record_preflop_action(self, opp_voluntarily_played):
        self.vpip_opps += 1
        if opp_voluntarily_played:
            self.vpip_events += 1
        self.redraw_opps += 1
 
    def record_fold_to_raise(self, folded):
        self.fold_3b_opps += 1
        if folded:
            self.fold_3b_events += 1
 
    def record_postflop_action(self, opp_was_aggressive):
        self.agg_opps += 1
        if opp_was_aggressive:
            self.agg_events += 1
 
    def record_cbet_response(self, opp_folded):
        self.cbet_fold_opps += 1
        if opp_folded:
            self.cbet_fold_events += 1
 
    def record_redraw(self, used_redraw):
        if used_redraw:
            self.redraw_events += 1
 
    def record_showdown(self, opp_hole_cards, went_to_showdown):
        self.wtsd_opps += 1
        if went_to_showdown:
            self.wtsd_events += 1
            if opp_hole_cards:
                key = _preflop_key(opp_hole_cards)
                self.showdown_hands.append(key)
 
    def end_hand(self):
        self.hands_played += 1
        self.this_hand = {}
 
    def summary(self):
        return (
            f"Hands={self.hands_played} | "
            f"VPIP={self.vpip:.2f} | "
            f"Fold3b={self.fold_to_3bet:.2f} | "
            f"FoldCbet={self.fold_to_cbet:.2f} | "
            f"Agg={self.aggression:.2f} | "
            f"WTSD={self.wtsd:.2f} | "
            f"Redraw={self.redraw_rate:.2f}"
        )
 
 
# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------
 
class Bot():
    '''
    Hybrid FSM + Monte Carlo + Opponent Modeling poker bot.
 
    Opponent model is built hand-by-hand in handle_round_over()
    and its adjustments are applied inside get_action().
    '''
 
    def __init__(self):
        self.opp       = OpponentModel()
        self.has_redrawn   = False
 
        # Per-hand scratch — track what happened so handle_round_over can update stats
        self._we_raised_preflop  = False
        self._we_cbet_flop       = False
        self._saw_flop           = False
        self._opp_redrawn        = False
 
    # ── handle_new_round ─────────────────────────────────────────────────────
 
    def handle_new_round(self, game_state, round_state, active):
        self.has_redrawn        = False
        self._we_raised_preflop = False
        self._we_cbet_flop      = False
        self._saw_flop          = False
        self._opp_redrawn       = False
 
    # ── handle_round_over ────────────────────────────────────────────────────
 
    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called at the end of every hand. We extract as much info as possible
        about what the opponent did and commit it to the opponent model.
 
        terminal_state fields we use:
          .previous_state  — the final RoundState before showdown
          .deltas          — [p0_delta, p1_delta] chip changes
        '''
        opp_idx    = 1 - active
        prev_state = terminal_state.previous_state   # final RoundState
        deltas     = terminal_state.deltas
 
        # Did the opponent voluntarily put chips in preflop?
        # Proxy: their pip > big blind (they did more than just post)
        opp_pip_preflop = prev_state.pips[opp_idx] if prev_state.street == 0 else BIG_BLIND
        opp_played_preflop = opp_pip_preflop > BIG_BLIND
        self.opp.record_preflop_action(opp_played_preflop)
 
        # Did they fold to our preflop raise?
        if self._we_raised_preflop:
            # If opp delta is negative and they folded, we won without showdown
            opp_folded_preflop = (
                deltas[active] > 0 and
                prev_state.street == 0 and
                prev_state.pips[opp_idx] < prev_state.pips[active]
            )
            self.opp.record_fold_to_raise(opp_folded_preflop)
 
        # Did they fold to our flop cbet?
        if self._we_cbet_flop and self._saw_flop:
            opp_folded_flop = (
                deltas[active] > 0 and
                prev_state.street <= 3
            )
            self.opp.record_cbet_response(opp_folded_flop)
 
        # Postflop aggression — did opp bet/raise on any street?
        if self._saw_flop:
            # Rough proxy: if we lost chips and didn't see a showdown, they bet us out
            # If they put in more than a call, they were aggressive
            final_pips = prev_state.pips
            opp_was_aggressive = final_pips[opp_idx] > final_pips[active]
            self.opp.record_postflop_action(opp_was_aggressive)
 
        # Redraw — did opponent use their redraw?
        self.opp.record_redraw(self._opp_redrawn)
 
        # Showdown — did we see their cards?
        opp_hole = prev_state.hands[opp_idx] if prev_state.hands[opp_idx] else None
        went_to_showdown = (
            opp_hole is not None and
            len(opp_hole) == 2 and
            prev_state.street == 5
        )
        self.opp.record_showdown(
            opp_hole if went_to_showdown else None,
            went_to_showdown
        )
 
        self.opp.end_hand()
 
    # ── get_action ───────────────────────────────────────────────────────────
 
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
 
        pot_odds = call_cost / (pot + call_cost) if (pot + call_cost) > 0 else 0.0
 
        # Track whether we saw the flop (for handle_round_over)
        if street >= 3:
            self._saw_flop = True
 
        # ── 2. Pull opponent model adjustments ──────────────────────────────
        #
        # These are small floats that nudge our thresholds based on
        # what we've learned about the opponent over previous hands.
        # If the model isn't reliable yet (< 15 hands), they all return 0/1.0.
 
        eq_adj        = self.opp.equity_call_adjustment()      # added to call threshold
        bluff_mult    = self.opp.bluff_size_multiplier()       # scales bluff bet size
        value_mult    = self.opp.value_size_multiplier()       # scales value bet size
        pf_raise_adj  = self.opp.preflop_raise_adjustment()    # extra chips on pf raise
        redraw_adj    = self.opp.redraw_threshold_adjustment() # lowers redraw threshold
 
        # Adjusted equity thresholds
        call_threshold_strong  = 0.55 + eq_adj   # need this much equity to call/bet
        call_threshold_thin    = 0.50 + eq_adj   # thin call threshold
        redraw_threshold       = max(0.01, 0.03 - redraw_adj)
 
        # ── 3. FSM — classify this decision ─────────────────────────────────
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
 
        # ── 4. TRIVIAL path ──────────────────────────────────────────────────
        if fsm == TRIVIAL:
            if street == 0 and _preflop_tier(hole) == 1:
                base = 3 * BIG_BLIND + pf_raise_adj   # raise bigger vs loose players
                amt  = min(max_r, max(min_r, base))
                if can_raise and amt >= min_r:
                    self._we_raised_preflop = True
                    return RaiseAction(amt)
                return CallAction() if can_call else CheckAction()
            return FoldAction() if can_fold else (CheckAction() if can_check else CallAction())
 
        # ── 5. STANDARD path ─────────────────────────────────────────────────
        if fsm == STANDARD:
            if street == 0:
                tier = _preflop_tier(hole)
                if tier == 2:
                    base = int(2.5 * BIG_BLIND) + pf_raise_adj
                    amt  = min(max_r, max(min_r, base))
                    # Raise more vs loose opponents, less vs tight
                    if can_raise and amt >= min_r and pot_odds < 0.20:
                        self._we_raised_preflop = True
                        return RaiseAction(amt)
                    if can_call and pot_odds < 0.25:
                        return CallAction()
                    return CheckAction() if can_check else FoldAction()
                if tier == 3:
                    if can_call and pot_odds < 0.15:
                        return CallAction()
                    return CheckAction() if can_check else FoldAction()
                return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())
 
            # Postflop standard — quick equity check
            eq = _mc_equity(hole, board, MC_FAST)
 
            if eq > 0.65:
                # Value bet — size up vs calling stations
                size = int(0.6 * pot * value_mult)
                amt  = min(max_r, max(min_r, size))
                if street == 3:   # track cbet
                    self._we_cbet_flop = True
                if can_raise and amt >= min_r:
                    return RaiseAction(amt)
                return CallAction() if can_call else CheckAction()
 
            if eq > call_threshold_strong:
                if can_call and call_cost > 0:
                    return CallAction()
                if can_check:
                    return CheckAction()
                # Small bet
                size = int(0.4 * pot * bluff_mult)
                amt  = min(max_r, max(min_r, size))
                if can_raise and amt >= min_r:
                    if street == 3:
                        self._we_cbet_flop = True
                    return RaiseAction(amt)
 
            return CheckAction() if can_check else (FoldAction() if can_fold else CallAction())
 
        # ── 6. COMPLEX path ───────────────────────────────────────────────────
        #
        # Step A  baseline equity
        # Step B  simulate every redraw candidate
        # Step C  pick base action from best equity + opponent model adjustments
        # Step D  wrap with RedrawAction if gain clears adjusted threshold
 
        # Step A — baseline
        base_eq = _mc_equity(hole, board, MC_FULL)
 
        # Step B — redraw search
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
                    if ms > os:  w += 1
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
                    if ms > os:  w += 1
                    elif ms == os: t += 1
                eq_swap = w / MC_DRAW + 0.5 * (t / MC_DRAW)
                if eq_swap > best_eq:
                    best_eq        = eq_swap
                    best_candidate = ('board', i)
 
        working_eq  = best_eq
        redraw_gain = best_eq - base_eq
 
        # Step C — equity → action (with opponent model adjustments applied)
        if working_eq > 0.70:
            # Strong hand — value bet, size up vs calling stations
            size = int((0.75 if street < 5 else 1.0) * pot * value_mult)
            amt  = min(max_r, max(min_r, size))
            if street == 3:
                self._we_cbet_flop = True
            if can_raise and amt >= min_r: base_action = RaiseAction(amt)
            elif can_call:                 base_action = CallAction()
            else:                          base_action = CheckAction()
 
        elif working_eq > call_threshold_strong:
            # Decent hand — semi-value or call
            size = int(0.45 * pot * value_mult)
            amt  = min(max_r, max(min_r, size))
            if can_raise and amt >= min_r and working_eq > pot_odds + 0.10:
                if street == 3:
                    self._we_cbet_flop = True
                base_action = RaiseAction(amt)
            elif can_call and call_cost > 0 and working_eq > pot_odds + eq_adj + 0.03:
                base_action = CallAction()
            elif can_check:
                base_action = CheckAction()
            else:
                base_action = FoldAction() if can_fold else CallAction()
 
        elif working_eq > pot_odds + eq_adj + 0.02:
            # Thin call — pot odds justify it after adjustment
            if can_call and call_cost > 0:
                base_action = CallAction()
            else:
                base_action = CheckAction() if can_check else FoldAction()
 
        elif working_eq < 0.38 and self.opp.fold_to_cbet > 0.55 and street == 3:
            # Bluff opportunity — we're weak but opponent folds to flops bets often
            size = int(0.5 * pot * bluff_mult)
            amt  = min(max_r, max(min_r, size))
            if can_raise and amt >= min_r:
                self._we_cbet_flop = True
                base_action = RaiseAction(amt)
            else:
                base_action = CheckAction() if can_check else FoldAction()
 
        else:
            base_action = CheckAction() if can_check else (FoldAction() if can_fold else CallAction())
 
        # Step D — attach redraw if gain clears the (model-adjusted) threshold
        if best_candidate is not None and redraw_gain >= redraw_threshold:
            self.has_redrawn = True
            ctype, idx = best_candidate
            return RedrawAction(base_action, ctype, idx)
 
        return base_action
