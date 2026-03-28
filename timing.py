import time


class TimeManager:
    def __init__(self, total_match_seconds=180.0, reserve_seconds=12.0):
        self.total_match_seconds = float(total_match_seconds)
        self.reserve_seconds = float(reserve_seconds)
        self.match_start = time.perf_counter()
        self.hand_start = None

    def start_hand(self):
        self.hand_start = time.perf_counter()

    def end_hand(self):
        self.hand_start = None

    def used_seconds(self):
        return time.perf_counter() - self.match_start

    def remaining_seconds(self):
        remaining = self.total_match_seconds - self.used_seconds()
        return max(0.0, remaining)

    def decision_budget(self, street, uncertain=False):
        """
        Returns a small time budget in seconds for this decision.
        Keeps a reserve so the bot does not time out late in the match.
        """

        remaining = self.remaining_seconds()
        usable = max(0.0, remaining - self.reserve_seconds)

        if usable <= 0.0:
            return 0.001

        if street == 0:      # preflop
            return min(0.004 if uncertain else 0.002, usable)

        if street == 3:      # flop
            return min(0.040 if uncertain else 0.020, usable)

        if street == 4:      # turn
            return min(0.060 if uncertain else 0.030, usable)

        if street == 5:      # river
            return min(0.020 if uncertain else 0.010, usable)

        return min(0.005, usable)
