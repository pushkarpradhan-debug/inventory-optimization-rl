# agents/rule_agent.py
# Rule-based agent — your existing simple_agent, as a clean class.
# This is the HUMAN-DESIGNED heuristic baseline.
#
# Core idea: project how much stock you'll have when the order arrives.
# If that projection is too low → order now, before the shortage happens.


class RuleBasedAgent:

    WORST_CASE_DAILY = 14   # conservative daily demand estimate (covers spikes)
    SAFETY_BUFFER    = 10   # extra cushion on top of lead-time coverage

    def act(self, obs) -> int:
        lead_time    = obs.lead_time
        target_stock = self.WORST_CASE_DAILY * lead_time + self.SAFETY_BUFFER

        # Projected stock when next order arrives
        projected = (
            obs.current_stock
            + obs.pending_order
            - self.WORST_CASE_DAILY * lead_time
        )

        # Order if projected stock will be too low
        if obs.pending_order == 0 and projected < self.SAFETY_BUFFER:
            return max(0, min(50, target_stock - obs.current_stock))

        # Order if stock is critically low right now
        if obs.pending_order == 0 and obs.current_stock < self.WORST_CASE_DAILY * 2:
            return max(0, min(50, target_stock - obs.current_stock))

        return 0

    def reset(self):
        pass
