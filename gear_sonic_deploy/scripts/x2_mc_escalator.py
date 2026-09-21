#!/usr/bin/env python3
"""x2_mc_escalator.py -- fire SetMcAction at high rate until accepted.

Purpose
-------

After ``deploy_x2.sh`` POSTs ``start_app`` on PC1, MC's services come
back online over a span of ~3-5 seconds. The slowest step in the
``HOLD_FOR_MC`` chain is the *first* successful ``SetMcAction(target)``
call from bash -- because each ``ros2 service call`` in bash spawns a
fresh Python process (rclpy init + service discovery), costing
~300 ms per attempt. Polling for "is MC up yet?" then making one
``ros2 service call`` adds up to ~700-800 ms of avoidable latency
between MC's first publish and the JOINT_DEFAULT mode entering.

This script avoids the per-call rclpy startup cost entirely. It
- ``rclpy.init()`` once,
- opens a persistent ``SetMcAction`` service client,
- fires the request at ``--rate-hz`` (default 20 Hz),
- exits the moment the response carries ``header.code == 0``,
- touches ``--success-sentinel`` so the calling bash script knows.

Failed calls before MC is up just return non-zero codes; we ignore
them and keep going. The expected timeline becomes:

::

  start_app POSTed
    ↓
  spawn escalator &                (fires at 20 Hz, mostly fails harmlessly)
    ↓
  ~3-5 s later, MC services up
    ↓
  next escalator tick succeeds      (sub-ms RTT; client is already connected)
    ↓
  success-sentinel touched         (≤50 ms after MC's first accepting tick)

Compared to today's "wait for MC, then call once from bash" path,
this is ~700 ms faster and removes the dual-publisher whir window
proportionally.

Usage from ``deploy_x2.sh``::

  python3 .../x2_mc_escalator.py \
      --target JOINT_DEFAULT \
      --rate-hz 20 \
      --timeout-s 30 \
      --success-sentinel /tmp/x2_mc_escalator_ok.<pid>.sentinel \
      --log /tmp/x2_mc_escalator.<pid>.log &

Exits 0 on success-sentinel, 1 on timeout, 2 on argparse error.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from aimdk_msgs.srv import (  # type: ignore[import-not-found]
    GetMcAction,
    SetMcAction,
)


SET_SERVICE_NAME = "/aimdk_5Fmsgs/srv/SetMcAction"
GET_SERVICE_NAME = "/aimdk_5Fmsgs/srv/GetMcAction"


def ts() -> str:
    """ms-precision wall-clock timestamp matching deploy_x2.sh's ts()."""
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"


class McEscalator(Node):
    """Holds open SetMcAction + GetMcAction clients.

    Persistent clients give us sub-ms RTT per call. Both services live
    on the same MC node, so once one is reachable the other is too.
    """

    def __init__(self, target_action: str, source_tag: str):
        super().__init__("x2_mc_escalator")
        self._set_client = self.create_client(SetMcAction, SET_SERVICE_NAME)
        self._get_client = self.create_client(GetMcAction, GET_SERVICE_NAME)
        self._target_action = target_action
        self._source_tag = source_tag

    def _wait_future(self, future, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            rclpy.spin_once(self, timeout_sec=min(0.005, remaining))
        return future.result()

    def set_action(self, call_timeout_s: float) -> int:
        """Send one SetMcAction(target). Returns header.code, or:
          -1: service not yet discovered
          -2: rpc timed out
          -3: rclpy returned no result
          -4: response missing expected fields
        """
        if not self._set_client.service_is_ready():
            return -1
        req = SetMcAction.Request()
        try:
            req.source = self._source_tag
        except AttributeError:
            pass
        try:
            req.command.action_desc = self._target_action
        except AttributeError:
            pass
        future = self._set_client.call_async(req)
        result = self._wait_future(future, timeout_s=call_timeout_s)
        if result is None:
            return -2
        try:
            return int(result.response.header.code)
        except AttributeError:
            return -4

    def get_action(self, call_timeout_s: float) -> str:
        """Send one GetMcAction. Returns the action_desc string MC reports
        (e.g. 'JOINT_DEFAULT', 'PASSIVE_DEFAULT'), or '' if the service is
        not yet up / response was malformed. Empty-string is the canonical
        'mode unknown / not yet ready' signal.
        """
        if not self._get_client.service_is_ready():
            return ""
        req = GetMcAction.Request()
        future = self._get_client.call_async(req)
        result = self._wait_future(future, timeout_s=call_timeout_s)
        if result is None:
            return ""
        try:
            return str(result.info.action_desc) or ""
        except AttributeError:
            return ""


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Hammer SetMcAction(target) until accepted; "
                    "touch a sentinel on success."
    )
    p.add_argument(
        "--target", default="JOINT_DEFAULT",
        help="MC mode to request via SetMcAction.command.action_desc."
    )
    p.add_argument(
        "--rate-hz", type=float, default=20.0,
        help="Retry rate when MC has not yet accepted. Default: 20 Hz "
             "(50 ms between attempts)."
    )
    p.add_argument(
        "--timeout-s", type=float, default=30.0,
        help="Give up after this many seconds without success. Default: 30."
    )
    p.add_argument(
        "--call-timeout-s", type=float, default=0.5,
        help="Per-attempt RPC deadline. Default: 0.5. Should be < 1 / rate-hz "
             "so we don't pile up calls if MC hangs."
    )
    p.add_argument(
        "--success-sentinel", required=True,
        help="Path to touch on first successful call (response.header.code==0)."
    )
    p.add_argument(
        "--source-tag", default="deploy_x2_escalator",
        help="String stamped into request.source for MC-side traceability."
    )
    p.add_argument(
        "--log", default="",
        help="Optional path to mirror stdout messages."
    )
    args = p.parse_args(argv)

    log_fh = None
    if args.log:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.log)) or ".",
                        exist_ok=True)
            log_fh = open(args.log, "w")
        except OSError:
            log_fh = None

    def log(msg: str) -> None:
        line = f"[{ts()}] [escalator] {msg}"
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()

    rclpy.init()
    rc = 1
    try:
        node = McEscalator(args.target, args.source_tag)
        log(f"started: target={args.target} rate={args.rate_hz}Hz "
            f"timeout={args.timeout_s}s sentinel={args.success_sentinel}")

        period = 1.0 / max(args.rate_hz, 1.0)
        deadline = time.monotonic() + args.timeout_s
        attempts = 0
        set_codes_seen: dict = {}
        get_modes_seen: dict = {}
        last_progress_log = time.monotonic()
        first_get_response_logged = False
        first_set_accepted_logged = False

        # The success criterion is GROUND-TRUTH: we require GetMcAction
        # to actually report the target mode. Earlier versions trusted
        # SetMcAction's response.header.code == 0, but in the cold-boot
        # window MC's set-service comes up *before* its mode-arbitration
        # is fully ready -- it accepts SetMcAction with code=0 and
        # SILENTLY ignores the request, leaving MC in PASSIVE_DEFAULT.
        # Verified the hard way on 2026-05-03 (run x2_run_20260503_213002):
        # escalator reported SUCCESS code=0 but recorder showed MC stayed
        # in PASSIVE_DEFAULT after deploy released the bus.
        #
        # Each tick:
        #   1. Send SetMcAction(target). Expect failures (-1, code != 0)
        #      until MC's services are alive. Once accepted, MC may or
        #      may not transition -- we don't trust the code.
        #   2. Call GetMcAction. If it reports our target mode, we're
        #      done. Otherwise loop.
        # We re-fire SetMcAction every tick so that whenever MC's
        # arbitration becomes ready, the next GetMcAction will reflect
        # the transition.
        while time.monotonic() < deadline:
            t_tick = time.monotonic()

            set_code = node.set_action(call_timeout_s=args.call_timeout_s)
            set_codes_seen[set_code] = set_codes_seen.get(set_code, 0) + 1
            if set_code == 0 and not first_set_accepted_logged:
                log(f"SetMcAction(target={args.target}) first accepted "
                    f"(code=0) at attempt {attempts + 1} "
                    f"(t+{(time.monotonic() - (deadline - args.timeout_s)):.3f}s)")
                first_set_accepted_logged = True

            current = node.get_action(call_timeout_s=args.call_timeout_s)
            if current and not first_get_response_logged:
                log(f"GetMcAction first response: mode={current!r}")
                first_get_response_logged = True
            if current:
                get_modes_seen[current] = get_modes_seen.get(current, 0) + 1

            attempts += 1

            if current == args.target:
                rtt_ms = (time.monotonic() - t_tick) * 1000.0
                log(f"GROUND-TRUTH SUCCESS: GetMcAction reports {current!r} "
                    f"after {attempts} attempts (tick RTT {rtt_ms:.1f}ms). "
                    f"set_codes={set_codes_seen} modes_seen={get_modes_seen}. "
                    f"Touching sentinel.")
                try:
                    os.makedirs(
                        os.path.dirname(os.path.abspath(args.success_sentinel))
                        or ".", exist_ok=True
                    )
                    with open(args.success_sentinel, "w") as fh:
                        fh.write(
                            f"target={args.target}\n"
                            f"confirmed_via=get_mc_action\n"
                            f"attempts={attempts}\n"
                            f"set_codes={set_codes_seen}\n"
                            f"modes_seen={get_modes_seen}\n"
                            f"ts={ts()}\n"
                        )
                    rc = 0
                except OSError as e:  # noqa: BLE001
                    log(f"FAILED to write success-sentinel: {e}")
                    rc = 3
                break

            # Periodic progress so the bash log shows we're alive.
            if time.monotonic() - last_progress_log >= 1.0:
                log(f"... {attempts} attempts so far. "
                    f"set_code last={set_code} all={set_codes_seen}. "
                    f"current_mode={current!r} all={get_modes_seen}. "
                    f"still trying ...")
                last_progress_log = time.monotonic()

            elapsed = time.monotonic() - t_tick
            sleep = period - elapsed
            if sleep > 0:
                time.sleep(sleep)

        if rc != 0:
            log(f"TIMEOUT after {attempts} attempts in {args.timeout_s}s. "
                f"Target {args.target!r} never confirmed via GetMcAction. "
                f"set_codes={set_codes_seen} modes_seen={get_modes_seen}.")
    finally:
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        if log_fh:
            log_fh.close()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
