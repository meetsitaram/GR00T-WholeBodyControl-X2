"""Pose-pipeline shared library.

Two processes share these helpers:

* :mod:`gear_sonic.scripts.x2_pose_mux` -- laptop-side N-to-1 pose
  multiplexer that arbitrates between the VLA bridge and the operator's
  override stream and publishes one merged wire to PC2.
* :mod:`gear_sonic_deploy.scripts.x2_pose_watchdog` -- PC2-side
  single-input fallback ladder (LIVE -> HOLD -> BLEND -> IDLE_CLIP) that
  keeps the C++ deploy's wire alive across a laptop death or wifi outage.

Splitting the original ``x2_pose_proxy.py`` into two processes that
share these helpers separates concerns:

* The merge logic (changes most) cannot crash the safety watchdog.
* The watchdog (smallest critical-path surface) stays on PC2 with the
  minimal numpy + pyzmq + stdlib dependency budget.
* The merge logic moves to the laptop where it's colocated with both
  source processes (no cross-host hops between bridge and merger).

Modules:

* :mod:`wire` -- packed message format constants + (de)coders. Shared
  by both processes since they both touch the wire.
* :mod:`clamp` -- numerically tiny per-element rate clamps. Used by
  the mux's engagement-ramp slow-step (and re-exported for tests of
  the bridge's tracking-feedback clamp).
* :mod:`fallback` -- staged HOLD/BLEND/IDLE_CLIP ladder + idle-clip
  replay + X2M2 loader. Only the watchdog uses these.
* :mod:`arbitrate` -- the stateful dual-source arbitration machine
  (engage gate, frozen detection, motion hysteresis, engagement
  slow-step ramp, edge event emitter). Only the mux uses this.
"""

from __future__ import annotations
