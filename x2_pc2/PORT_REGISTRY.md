# PORT REGISTRY — single source of truth for every ZMQ port in the stack

**Rule: no new socket without a row here first.** This file exists because of
the 2026-08-30 battery-pull incident: the whole-body token service was given
:5556 based on the SIM port map, but on the robot :5556 is the kplanner's
pose PUB — the service grabbed it first at ignition, the kplanner crashed on
bind, and the pad chain INCLUDING THE PAD E-STOP went dead. The port was
picked without auditing the robot map; this registry makes that audit a
30-second lookup.

The sim stack (`simstack_local.sh`) deliberately renames the core chain
(+100) for host-network isolation; every other port is the SAME number on
both hosts. When adding a port, check BOTH columns.

| port | robot (PC2)                                    | sim workstation                | owner / notes |
|------|------------------------------------------------|--------------------------------|---------------|
| 5513 | eval tooling (see grep)                        | same                           | misc eval scripts |
| 5519 | eval tooling                                   | same                           | misc eval scripts |
| 5534 | eval tooling                                   | same                           | misc eval scripts |
| 5555 | camera stream                                  | same                           | record_x2_dataset.py |
| 5556 | **kplanner pose PUB** (head of pad chain)      | *(sim uses 5656)*              | pc2_kplanner_onnx.py; the 2026-08-30 collision victim |
| 5557 | deploy x2_debug PUB                            | *(sim uses 5659)*              | x2_deploy_onnx_ref |
| 5558 | watchdog downstream pose (deploy VLA input)    | *(sim uses 5658)*              | x2_pose_watchdog.py |
| 5559 | scene_state SUB                                | same                           | record_x2_dataset.py |
| 5560 | scene_reset PUB                                | same                           | record_x2_dataset.py |
| 5563 | planner_cmd (pad bridge / quest manager → kplanner) | *(sim uses 5663)*         | pad_locomotion_bridge.py, quest3_manager_x2.py |
| 5564 | arm_targets + hands                            | same                           | quest stack / recorder |
| 5565 | body pose sub                                  | same                           | record_x2_dataset.py |
| 5566 | SAFE_IDLE pose-resume chord PUB                | same                           | quest stack RESUME_PUB_PORT |
| 5567 | motor monitor                                  | same                           | quest3_manager_x2.py |
| 5568 | motion_clip_cmd (dances/gestures → kplanner)   | *(sim uses 5668)*              | play_xbox_controller.py, pad bridge clips |
| 5569 | pad daemon pad_state PUB                       | (pad daemon not run in sim)    | pc2_pad_daemon.py — why intent moved to 5573 |
| 5570 | pad rumble PULL                                | bridge robot_pose PUB (sim-only ground truth) | different hosts, no conflict — but never merge them |
| 5571 | kplanner lidar-guard SUB                       | same                           | _GUARD_PORT, fail-open |
| 5572 | kplanner arm-ingest SUB bind                   | same                           | arm_targets/hand_finger_cmd |
| 5573 | **pico intent** (laptop → token service SUB bind) | same                        | whole-body teleop; moved from 5569 (pad daemon collision) |
| 5574 | **motion tokens** (token service PUB → deploy) | same                           | whole-body teleop; moved from 5556 (KPLANNER COLLISION — the incident) |
| 5575–5579 | — FREE —                                  | — FREE —                       | next allocations start here |
| 5580 | launch_inference PUB                           | same                           | launch_inference.py |
| 5581–5582 | — FREE — *(reserved: Isaac Lab bridge, not shipped)* | — FREE —            | keep clear |
| 5656 | *(robot uses 5556)*                            | kplanner pose PUB              | simstack isolated map |
| 5657 | *(robot: n/a)*                                 | kplanner's own debug PUB       | simstack; deploy debug moved OFF this (fall #2) |
| 5658 | *(robot uses 5558)*                            | watchdog downstream pose       | simstack |
| 5659 | *(robot uses 5557)*                            | deploy x2_debug PUB            | simstack |
| 5663 | *(robot uses 5563)*                            | planner_cmd                    | simstack |
| 5668 | *(robot uses 5568)*                            | motion_clip_cmd                | simstack |

Laptop-side listeners: XRoboToolkit PC Service gRPC :60061 (Pico headset link).
