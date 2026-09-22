# X2 stack architecture

Diagrams of the runtime stacks, the teleop sources, the model build chain and
the training loop. Port numbers are the robot map from
[`x2_pc2/PORT_REGISTRY.md`](../../x2_pc2/PORT_REGISTRY.md); the sim map is the
same chain renamed +100 (5556 -> 5656, 5563 -> 5663, 5568 -> 5668).

## 1. Runtime pose chain (sim and robot share it)

Every controller ends in one place: a 50 Hz reference pose (or a motion
token) into the SONIC deploy node, which runs the ONNX policy and drives the
plant. In sim the plant is MuJoCo behind `x2_mujoco_ros_bridge.py`; on the
robot it is the AgiBot SDK behind the same colcon package.

```mermaid
flowchart LR
  subgraph sources["Command sources"]
    pad["Gamepad<br/>play_xbox_controller.py / pc2_pad_daemon.py"]
    quest["Quest 3 headset<br/>quest3_manager_x2.py"]
    pico["Pico + trackers<br/>live_pico_smpl_teleop.py"]
    clip["Motion clip<br/>run_motion_replay.sh, motion_clip_session.py"]
  end
  pad -->|"planner_cmd :5563"| bridge["pad_locomotion_bridge.py"]
  bridge -->|"planner_cmd :5563"| kp["kplanner ONNX<br/>pc2_kplanner_onnx.py"]
  quest -->|"planner_cmd :5563 + arm_targets :5564"| kp
  clip -->|"motion_clip_cmd :5568"| kp
  kp -->|"pose PUB :5556"| merger["x2_pose_merger.py<br/>(gestures, arm overlay)"]
  merger --> wd["x2_pose_watchdog.py<br/>tilt / SAFE_IDLE :5566"]
  wd -->|"pose :5558"| deploy["SONIC deploy node<br/>x2_deploy_onnx_ref (colcon)"]
  pico -->|"pico intent :5573"| tok["token service<br/>smpl tokenizer ONNX"]
  tok -->|"motion tokens :5574"| deploy
  deploy -->|"joint cmds"| plant["Plant<br/>MuJoCo bridge (sim) / AgiBot SDK (robot)"]
  plant -->|"x2_debug :5557"| deploy
```

The kplanner side is the pose encoder path of the policy; the Pico side is
the SMPL encoder path. `x2_pose_merger.py` arbitrates between them and the
gesture catalog (F08: one operator toggles locomotion and whole-body follow).

## 2. Sim launchers

```mermaid
flowchart TB
  sop["sim_onnx_planner.sh<br/>(one-shot: pad / --vr-only / --whole-body-teleop)"]
  sl["simstack_local.sh<br/>(persistent, docker x2sim)"]
  sop --> sl
  sl --> dx["deploy_x2.sh sim<br/>colcon build + deploy node in docker"]
  sl --> kp["pc2_kplanner_onnx.py :5656"]
  sl --> wd["x2_pose_watchdog.py :5658"]
  sl --> pb["pad_locomotion_bridge.py"]
  sl --> ts["token service (whole-body)"]
  sl --> mv["sim_mirror_viewer.py<br/>host MuJoCo viewer"]
  dx --> mj["x2_mujoco_ros_bridge.py<br/>MuJoCo plant, sim_init_poses.yaml"]
  q["run_x2_quest3_planner_stack.sh"] --> sl
  q --> qm["quest3_manager_x2.py"]
  q --> pm["x2_pose_merger.py"]
  p["run_x2_pico_wbc.sh / run_pico_replay.sh"] --> sl
  p --> lp["live_pico_smpl_teleop.py<br/>(headset or --tape-replay)"]
```

## 3. Robot bring-up (PC2, colcon)

```mermaid
flowchart LR
  dev["Workstation clone"] -->|"pc2_bringup.sh: rsync + pip + colcon"| pc2["PC2 (Orin NX)<br/>$PC2_PREFIX"]
  dev -->|"push_to_pc2.sh: models + env, 4 gates, md5"| pc2
  pc2 --> ritual["ritual_start_sonic.sh (boot daemon)<br/>robot_env.env: model, tuning, plant"]
  ritual --> daemons["x2_pc2_daemons.sh<br/>kplanner, watchdog, pad daemon, token service"]
  ritual --> node["x2_deploy_onnx_ref<br/>trained_gains_s0_inc_waistmc.yaml"]
  node --> sdk["AgiBot SDK<br/>MC handover: x2_stand_default_pose.yaml"]
```

## 4. Model build chain

```mermaid
flowchart LR
  ckpt["Training checkpoint<br/>last.pt + config.yaml"]
  ckpt -->|"export_native_all.sh"| native["native set<br/>_dual / _g1 / _g1_token / _smpl_tokenizer"]
  ckpt -->|"frozen_core_t2_export.py (fold LoRA)<br/>+ token export"| frozen["frozen G1-core set<br/>_g1 / _g1_token + v11 tokenizer"]
  native & frozen -->|"onnx_provenance.py check"| gate{"codec fingerprint<br/>pairing OK?"}
  gate --> models["gear_sonic_deploy/models/ (shipped)<br/>or $X2_MODELS (bring your own)"]
  planner["kplanner torch ckpts<br/>(HF kplanner_torch/)"] -->|"reexport_x2_onnx.sh"| pl["planner graphs<br/>PLANNER_MODEL"]
  mc["MC stock gestures<br/>x2_recorded/mc_gestures/*.pkl"] -->|"build_demo_bank_from_upstream.sh"| bank["x2_demo_bank.pkl + x2m2<br/>(gitignored, regenerated)"]
  bank --> pad["pad gesture bank (L1 + A/B/X/Y)"]
  models --> sim["sim launchers"] & robot["push_to_pc2.sh"]
```

Which artifact rebuilds when a checkpoint or a script changes is tabulated in
[`BUILD_CHAIN.md`](BUILD_CHAIN.md).

## 5. Training loop (Isaac Lab, Nebius)

```mermaid
flowchart LR
  chores["Reference motions<br/>x2_pico_chores_0917.pkl + smpl_sidecars/"]
  chores -->|"check_smpl_sidecars.py<br/>preflight_train.py"| gates{"gates PASS?"}
  gates -->|"run_smoke_8gpu.sh"| smoke["sonic_x2_ultra_smoke<br/>200 iterations"]
  gates -->|"launch_bigrun.sh / run_pico_v8_8gpu.sh"| run["sonic_x2_ultra /<br/>sonic_x2_leg5_pico_combined_ft"]
  g1["nvidia/GEAR-SONIC G1 core"] -->|"make_g1_lora_warmstart.py"| lora["sonic_x2_frozen_core_t2 /<br/>s1_fresh_vendor_waistmap (LoRA)"]
  smoke & run & lora --> ckpt["checkpoint + config.yaml"]
  ckpt --> export["build chain (section 4)"]
```

Three encoders train together on the same corpus: the G1 pose encoder, the
teleop encoder and the SMPL encoder (`sonic_x2_ultra.yaml`); every smoke and
training run therefore needs the SMPL sidecars next to the retargeted clips.

## 6. Training infrastructure (as used for the shipped sets)

![X2 training topology: four 8 x H100 nodes on one InfiniBand fabric with node-local NVMe, a shared filesystem, a 1 x H100 eval probe, the workstation running the MuJoCo twin, and the X2 robot](../../media/x2/x2_training_topology.png)

<details>
<summary>Text version of the diagram</summary>

```mermaid
flowchart LR
  subgraph nebius["Nebius cloud"]
    direction LR
    subgraph cluster["Training cluster: 4 x 8 H100 = 32 GPUs (accelerate, rank 0 = rendezvous)"]
      direction TB
      n0["node 0<br/>8 x H100, NVMe: corpus + sidecars + run dir"]
      n1["node 1<br/>8 x H100, NVMe: corpus + sidecars"]
      n2["node 2<br/>8 x H100, NVMe"]
      n3["node 3<br/>8 x H100, NVMe"]
      n0 --- n1 --- n2 --- n3
    end
    share[("Shared filesystem<br/>corpus + SMPL sidecars (source of truth)<br/>checkpoints every 10 min, launch records")]
    probe["Probe: 1 x H100<br/>Isaac Lab milestone evals"]
  end
  subgraph lab["Lab"]
    ws["Workstation<br/>export chain -> ONNX<br/>MuJoCo sim stack (deploy binary in docker)"]
    pc2["AgiBot X2 Ultra<br/>PC2 runs the same deploy node"]
  end
  share -- "stage_local.sh -> node-local NVMe" --> n0
  n0 -- "rank-0 checkpoints, rsync 10 min" --> share
  share -- "milestones" --> probe
  probe -- "eval results" --> share
  share -- "milestone .pt + config.yaml" --> ws
  ws -- "push_to_pc2.sh (ONNX + env, md5 gates)" --> pc2
  pc2 -- "Pico tapes, black-box logs" --> ws
  ws -- "new corpus pieces + sidecars" --> share
```

</details>

Training reads 100k+ small files from 32 processes, so `stage_local.sh` copies
the corpus to each node's NVMe before launch and only rank 0 writes
checkpoints, rsynced to the share every 10 minutes. The probe scores each
milestone without touching the training nodes; the workstation qualifies an
exported set in the deploy-faithful MuJoCo stack before `push_to_pc2.sh` puts
it on the robot. See [`F12_training_nebius.md`](F12_training_nebius.md) and
[`TRAINING_NOTES.md`](TRAINING_NOTES.md).
