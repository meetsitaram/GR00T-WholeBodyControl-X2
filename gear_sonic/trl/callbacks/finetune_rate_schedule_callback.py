import math

from transformers import TrainerCallback


class FinetuneRateScheduleCallback(TrainerCallback):
    """Anneal ``fine_tune_dataset.finetune_sample_rate`` during training.

    WHY: the rate is read out of ``m_cfg`` on every call to
    ``MotionLibBase._apply_finetune_sample_rate``, but nothing ever writes it, so
    it is effectively fixed at launch. Changing it therefore costs a RESTART --
    and a restart is what produces the naive-adaptive-sampler transient that
    makes the following ~1-2k iterations unreadable. Stepping a drill down by
    hand (0.5 -> 0.3 -> 0.15) buys three of those. This callback does the same
    taper with none, and lets the fine-tune pressure decay in step with the LR
    cosine instead of fighting it.

    Mutating ``m_cfg`` is safe and needs no cross-rank sync: the schedule is a
    pure function of ``state.global_step``, so every rank computes the identical
    rate on the identical step. Bin counts are untouched, so a resumed run still
    restores its adaptive-sampling history (that guard compares only the total
    bin count, which comes from the corpus, not from this rate).

    KNOWN LIMITATION -- read before relying on this. The fine-tune route samples
    UNIFORMLY over the pinned clips, bypassing the adaptive sampler, so a single
    global rate shrinks a mastered clip and a failing clip by exactly the same
    factor. It does NOT "decay once learned"; it decays everything on a clock.
    The per-clip version of this idea (weight fine-tune draws by failure rate, or
    route them through the adaptive sampler) is the real fix and would make most
    of this schedule unnecessary. Prefer a `final_rate` floor well above zero:
    the low-LR tail is where consolidation happens, and starving the target
    clips there invites drift back toward the breadth distribution.

    Args:
        start_rate: rate held until ``start_step``.
        final_rate: floor held from ``end_step`` on. Keep it > 0.
        start_step: absolute global step at which the decay begins. Absolute,
            not relative, so it composes with a resumed run's step counter.
        end_step: absolute global step at which ``final_rate`` is reached.
        schedule: ``"cosine"`` (default) or ``"linear"``.
        log_every: print the active rate every N steps (0 disables).
    """

    def __init__(
        self,
        start_rate,
        final_rate,
        start_step,
        end_step,
        schedule="cosine",
        log_every=250,
    ):
        super().__init__()
        if end_step <= start_step:
            raise ValueError(f"end_step ({end_step}) must exceed start_step ({start_step})")
        if schedule not in ("cosine", "linear"):
            raise ValueError(f"unknown schedule '{schedule}' (expected cosine or linear)")
        if final_rate > start_rate:
            raise ValueError(
                f"final_rate ({final_rate}) > start_rate ({start_rate}); this callback anneals down"
            )
        self.start_rate = float(start_rate)
        self.final_rate = float(final_rate)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.schedule = schedule
        self.log_every = int(log_every)
        self._warned = False

    def rate_at(self, step):
        """The scheduled rate at an absolute global step. Pure; safe to unit-test."""
        if step <= self.start_step:
            return self.start_rate
        if step >= self.end_step:
            return self.final_rate
        t = (step - self.start_step) / (self.end_step - self.start_step)
        if self.schedule == "linear":
            decay = 1.0 - t
        else:  # cosine
            decay = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.final_rate + (self.start_rate - self.final_rate) * decay

    def on_step_end(self, args, state, control, **kwargs):
        env = kwargs.get("env")
        motion_lib = getattr(env, "_motion_lib", None)
        m_cfg = getattr(motion_lib, "m_cfg", None)
        ft_cfg = m_cfg.get("fine_tune_dataset", None) if m_cfg is not None else None

        if not ft_cfg or not ft_cfg.get("enable", False):
            if not self._warned:
                self._warned = True
                print(  # noqa: T201
                    "FinetuneRateScheduleCallback: fine_tune_dataset is absent or disabled — "
                    "no rate to schedule, callback is a no-op."
                )
            return

        rate = self.rate_at(state.global_step)
        ft_cfg["finetune_sample_rate"] = rate

        if (
            self.log_every
            and state.global_step % self.log_every == 0
            and getattr(state, "is_world_process_zero", True)
        ):
            print(  # noqa: T201
                f"[finetune_rate_schedule] step {state.global_step}: "
                f"finetune_sample_rate = {rate:.4f}"
            )
