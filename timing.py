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
        return max(0.0, self.total_match_seconds - self.used_seconds())
if not uncertain else 0.040, usable)
        if street == 4:
            return min(0.030 if not uncertain else 0.060, usable)
        if street ==
    def decision_budget(self, street, uncertain=False):
        remaining = self.remaining_seconds()
        usable = max(0.0, remaining - self.reserve_seconds)
        if usable <= 0.0:
            return 0.001

        if street == 0:
            return 0.002 if not uncertain else 0.004
        if street == 3:
            return min(0 5:
            return min(0.010 if not uncertain else 0.020, usable)
        return 0.005