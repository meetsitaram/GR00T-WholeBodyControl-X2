"""Heuristic locomotion planner support library for AgiBot X2 Ultra.

Submodules:
  - constants : 31-DOF MuJoCo joint order, waist/leg/arm indices, default pose.
  - metrics   : per-clip / per-window metrics used by the curator
                (net XY, net yaw, end-at-square, feet-planted, etc.).
  - blending  : SLERP / LERP / yaw-cylinder helpers lifted from
                the (unshipped) playlist stitcher so the curator
                and the runtime planner share the same math.
  - registry  : bin-spec / primitive-registry YAML I/O.
  - x2_recipes : recipe DSL for building X2 primitives (clip_window /
                 synthesize_* ops + transforms). Embodiment-specific.
  - state_machine : runtime state machine used by ``x2_heuristic_planner.py``.

See ``docs/source/references/x2_heuristic_planner.md`` for the higher-level
design.
"""
