# X2 gamepad cheat sheet (demo pad, untethered)

Bindings come from two places and the sheet reflects their combination:
`x2_pc2/pad_bindings.env` (deadman, lock-speed, the bank ENABLE lists) as
consumed by `x2_pc2/ritual_start_demo.sh` on the robot and by
`gear_sonic/scripts/simstack_local.sh` in sim, and
`gear_sonic/scripts/pad_locomotion_bridge.py` (deadman handling, e-stop
gesture, bank-button map, the L1+R1 STOP chord, `--lock-speed`). Xbox names
below; the bridge auto-detects a DualSense (Cross=A, Circle=B, Square=X,
Triangle=Y). A printable map is `gamepad_control_map.html` in this directory.

## Ignition (robot only)
| action | input |
|---|---|
| Ignite stack | hold **L1+L2+R1+R2** ~3 s (haptic countdown) -> press **Y** |
| Stop a playing dance/clip | **L1+R1** chord (-> idle, weak rumble blip) |

## Driving (DEADMAN = hold **L2 only**, `PAD_DEADMAN=left`; one-handed, the right hand spots the robot)
| action | input |
|---|---|
| Walk / steer | left stick while holding L2; right stick X = yaw while walking. Yaw is **continuous**: 25 % of the arc rate at the deadzone edge, full at full deflection (`KPLANNER_YAW_PROPORTIONAL=0` = old fixed turn; F01) |
| Speed | **fixed** (`PAD_LOCK_SPEED=1`): launch value from `KPLANNER_FIXED_FWD_MPS` in the profile. No in-drive nudges (unset `PAD_LOCK_SPEED` to re-enable L1/R1 -0.1/+0.1) |
| Obstacle guard (if armed) | inside 1.6 m fwd/lat gets clamped and **latches**; yaw always works — turn away, then **release + re-hold L2** to reset |
| Stop walking | release L2 (robot idles) |
| Crouch mode | hold **L2+R2** + D-pad DOWN = arm; crouched only while both triggers stay held; release either = standing. Needs the **G1-core planner** (`PLANNER_MODEL` from HF `tinkerbuggy/sonic-x2/kplanner_g1core/`); the incumbent template planner has no crouch template and ignores the hip. Robot-verified 2026-09-09 (frozen core); depth varies with the sampler seed (`KPLANNER_CROUCH_SEED`). See F01 "Crouch mode" |

## E-STOP (any time)
1. Hold **A+X together** (chord arms — the bridge logs it), then
2. **Pump both triggers** (LT/RT):
   - first phase = **SOFT stop** — abort to idle stand;
   - **keep pumping ~1 s more** -> **PURE DAMPING** (full damp).
- Robot in a deep crouch / tilted low: go straight through to full damping.
- MC stand-mode refusals while the robot is down are expected — never fight them.

## Clip banks (L2 RELEASED; same chord again = next clip in bank; **L1+R1 = STOP**)

Shipped defaults: only the **gesture bank (L1+A)** is bound, to the X2 Motion-Controller stock gestures recorded on the robot (`gear_sonic/data/motions/x2_recorded/mc_gestures/`, 51 clips; default slots: right wave, kiss, high-five, shake, turn-wave right/left). The dance, combat, medium and turn banks ship **empty**: fill them in `x2_pc2/pad_bindings.env` with keys of clips you place in the dances dir (no third-party motion data is bundled).
| chord | bank | env list |
|---|---|---|
| **L1+Y** | easy dances | `EASY_DANCES` |
| **L1+X** | combat | `COMBAT` |
| **L1+B** | medium dances | `MEDIUM` |
| **L1+A** | gestures | `GESTURES` |

Bank contents are the keys listed in `x2_pc2/pad_bindings.env`; each key must
exist as an `.x2m2` in the dances dir (`$CKPT_ROOT/dances_x2m2/` in sim,
`${PC2_PREFIX}/planner_stack/models/dances_x2m2/` on the robot). With the
bank built by `tools/build_demo_bank_from_upstream.sh` the keys are the gesture
file stems (e.g. `right_wave_001`); upstream reference clips only appear with
`--with-upstream-examples`.

## Fixed clips (L2 RELEASED)
| input | clip | note |
|---|---|---|
| Right stick LEFT / RIGHT held | in-place turn clip (`TURNS` list) | fires on entry; test with a spotter |
| D-pad UP | `PAD_CLIP_DPAD_UP` | empty = disabled |
| RT held + left stick (8-way) | micro-step primitives (`MICRO` list, `prim:<name>` = kplanner primitive) | empty slot = unbound |

## Rules of thumb
- One bank per face button: **Y=easy X=combat A=gestures B=medium**.
- Dancing and driving never mix: clips fire only with L2 released.
- Combat clips travel ~1.3 m — leave floor room.
- Never kill planner/deploy processes mid-motion; use the e-stop gesture.

## Regenerating the printable map

`gamepad_control_map.html` is self-contained (one inline SVG). Edit the SVG
labels together with this sheet, then rasterise:

```bash
.venv/bin/python - <<'PY'
import re, cairosvg
h = open('docs/x2/gamepad_control_map.html').read()
svg = re.search(r'<svg.*</svg>', h, re.S).group(0)
svg = svg.replace('xmlns="http://www.w3.org/2000/svg">',
                  'xmlns="http://www.w3.org/2000/svg"><rect width="1000" height="700" fill="#fff"/>', 1)
open('/tmp/padmap.svg','w').write(svg)
cairosvg.svg2png(url='/tmp/padmap.svg', write_to='/tmp/gamepad_control_map.png', output_width=3300, output_height=2310)
PY
```

Print the HTML from a browser in landscape, or the PNG.
