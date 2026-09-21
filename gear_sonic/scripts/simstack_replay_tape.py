#!/usr/bin/env python3
"""Replay a kplanner intent tape into the LIVE local sim stack
(gear_sonic/scripts/simstack_local.sh).

Why this and not ``replay_intents_to_daemon.py``: that tool launches its own
planner on private ports and reads the planner's tape back -- it proves what the
REFERENCE did, with no physics. §24 needs the whole loop (planner -> deploy ->
MuJoCo) so the question "does the reference make the ROBOT burst" can be asked
at all. Same tape format, same recorded timings; only the destination differs.

The operator's commands are replayed at their recorded inter-event deltas, so a
stick slammed to full in 20 ms is replayed as 20 ms -- which is the whole point:
the 2026-08-25 burst followed a full-speed walk, a 4 s pause, then a hard yaw
step, and none of that survives a hand-typed approximation.

  usage: simstack_replay_tape.py --tape T.jsonl [--from -20] [--to 0]
                                 [--estop-wall 1787636422.066] [--speed 1.0]
"""
from __future__ import annotations
import argparse, json, sys, time
import zmq


def load_intents(path):
    out = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("ev") == "intent_recv":
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", required=True)
    ap.add_argument("--port", type=int, default=5663)
    ap.add_argument("--topic", default="planner_cmd")
    ap.add_argument("--estop-wall", type=float, default=None,
                    help="wall clock of the e-stop; --from/--to are relative to it")
    ap.add_argument("--from", dest="t_from", type=float, default=None)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ev = load_intents(a.tape)
    if not ev:
        print("  no intent_recv events in tape", file=sys.stderr)
        return 2

    def tw(d):
        return d.get("tw") or d.get("tm")

    if a.estop_wall is not None:
        ref = a.estop_wall
        if a.t_from is not None:
            ev = [d for d in ev if tw(d) - ref >= a.t_from]
        if a.t_to is not None:
            ev = [d for d in ev if tw(d) - ref <= a.t_to]
    if not ev:
        print("  window selected no events", file=sys.stderr)
        return 2

    t0 = tw(ev[0])
    print(f"  replaying {len(ev)} intents over {tw(ev[-1]) - t0:.2f}s "
          f"(speed {a.speed}x) -> tcp://127.0.0.1:{a.port}")

    sock = None
    if not a.dry_run:
        sock = zmq.Context.instance().socket(zmq.PUB)
        sock.connect(f"tcp://127.0.0.1:{a.port}")
        time.sleep(0.4)          # PUB-connect needs a beat before the first send

    wall0 = time.monotonic()
    last = None
    for d in ev:
        due = wall0 + (tw(d) - t0) / a.speed
        gap = due - time.monotonic()
        if gap > 0:
            time.sleep(gap)
        payload = {k: v for k, v in d.items()
                   if k not in ("ev", "tm", "tw", "wall")}
        if sock is not None:
            sock.send_multipart([a.topic.encode("ascii"),
                                 json.dumps(payload).encode("utf-8")])
        cur = (payload.get("stick_fwd"), payload.get("stick_side"),
               payload.get("stick_yaw"))
        if cur != last:
            rel = tw(d) - (a.estop_wall if a.estop_wall else t0)
            print(f"  t={rel:+8.3f}  fwd={cur[0]:+.2f} side={cur[1]:+.2f} "
                  f"yaw={cur[2]:+.2f}", flush=True)
            last = cur
    time.sleep(0.3)
    print("  replay complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
