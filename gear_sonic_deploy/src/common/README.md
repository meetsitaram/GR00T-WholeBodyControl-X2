# `sonic_common` — robot-agnostic deploy-time utilities

This directory is the long-term home for C++ headers and source files that
are shared across deploy targets (currently G1 and X2 Ultra; future robots
welcome). It exists because two "would-be portable" categories of code
were originally written inside the G1 deploy package:

1. Pure utilities (numpy IO, XML parser, math, generic FK over MJCF, ONNX
   Runtime wrapper, file logging, ring-buffer state logger).
2. Patterns that *should* be portable but ended up coupled to G1
   constants (motion-data reader baked in `policy_parameters.hpp`, etc.).

The point of `sonic_common` is to grow category (1) into a real shared
library and to refactor category (2) into robot-agnostic templates +
robot-specific configs.

---

## Phase 2 status (current)

Phase 2 is intentionally **non-disruptive**: zero G1 source files were
modified. The G1 build is byte-for-byte unchanged.

What Phase 2 actually delivers:

- **Audit of the G1 header tree** (this file, "Curated portable subset"
  below).
- **CMake seam** — `sonic_common` `INTERFACE` library declared in
  `CMakeLists.txt`. It re-exports the curated portable headers under a
  stable name so Phase 3 (X2 deploy) can `target_link_libraries(...
  sonic_common)` and `#include "fk.hpp"` without poking into
  `src/g1/g1_deploy_onnx_ref/include/` directly.
- **Decoupling rule** documented (this file, "Migration plan" below).

Phase 2 deliberately does **not** physically move files. That move
happens after X2 bring-up has confirmed which utilities truly are
robot-agnostic in practice. Doing the move now would touch every
`#include "..."` line in the active G1 build — a regression risk we are
choosing not to take until we have an X2 binary linking the same headers
as the regression check.

---

## Curated portable subset (audited 2026-04-21)

The following headers in `src/g1/g1_deploy_onnx_ref/include/` were
inspected and found to have **no G1-specific compile-time dependency**.
Phase 3 X2 code may include them via the `sonic_common` include path:

| Header              | LOC  | Hard deps                  | Notes                                                                |
|---------------------|------|----------------------------|----------------------------------------------------------------------|
| `cnpy.h`            | 269  | `<zlib.h>` + stdlib        | Numpy `.npy/.npz` reader/writer.                                     |
| `xml.h`             | 503  | stdlib                     | Header-only XML parser (vendored, MIT).                              |
| `math_utils.hpp`    | 506  | stdlib + Eigen (header)    | Quaternion / rotation helpers, `float_to_double`, etc.               |
| `fk.hpp` / `.cpp`   | 58   | `xml.h`                    | Generic MJCF forward kinematics — accepts any MuJoCo XML.            |
| `ort_session.hpp`   | 204  | ONNX Runtime               | Thin RAII wrapper. Currently unused by g1; resurrected for X2.       |
| `file_sink.hpp`/`.cpp` | 51 | stdlib                     | Atomic CSV file writer.                                              |
| `state_logger.hpp`/`.cpp` | 273 | `file_sink.hpp` + stdlib | Ring-buffered CSV logger; `num_joints` is a runtime parameter.       |
| `utils.hpp`         | 297  | stdlib                     | `DataBuffer<T>` template, time helpers, generic concurrency utils.   |

**Companion source files** (live in `src/g1/g1_deploy_onnx_ref/src/`):
`cnpy.cpp`, `fk.cpp`, `file_sink.cpp`, `state_logger.cpp`. The G1 package
already compiles them via `file(GLOB_RECURSE ...)`. Phase 3 will compile
its own translation units via the X2 package's source list (separate
static archive — no ODR conflict).

---

## NOT portable (G1-coupled, must not be linked from non-G1 targets)

| Header                        | Coupling                                                        |
|-------------------------------|------------------------------------------------------------------|
| `policy_parameters.hpp`       | 29-DOF G1 motor types and joint maps. X2 has its own (Phase 1). |
| `robot_parameters.hpp`        | `G1_NUM_MOTOR`, joint name strings.                              |
| `error_monitor.hpp`           | Iterates over `G1_NUM_MOTOR` motor states.                       |
| `dex3_hands.hpp`              | Unitree Dex3 hand SDK types.                                     |
| `motion_data_reader.hpp`      | Pulls in `policy_parameters.hpp` + `fk.hpp` for G1 motion PKLs.  |
| `observation_config.hpp`      | G1 obs layout (29-DOF widths).                                   |
| `encoder.hpp`                 | TensorRT engine signatures keyed to G1 latent sizes.             |
| `control_policy.hpp`          | G1 PD loop + policy step orchestration.                          |
| `localmotion_kplanner*.hpp`   | G1 kinematic planner with G1-specific obs builders.              |
| `input_interface/*`, `output_interface/*` | Mostly G1-coupled (rely on `policy_parameters.hpp`); X2 will get its own AimDK ROS 2 IO layer. |

X2's equivalents will be authored from scratch in Phase 3 inside
`src/x2/agi_x2_deploy_onnx_ref/`, drawing on `sonic_common` for utilities
and on the auto-generated `x2/.../include/policy_parameters.hpp`
(Phase 1) for robot constants.

---

## Migration plan (post X2 bring-up)

After X2 reaches a known-good standing test on real hardware (Phase 6),
do the actual file relocation in a single dedicated PR:

1. Move each file in the "Curated portable subset" table from
   `src/g1/g1_deploy_onnx_ref/include/` (and `.../src/`) to
   `src/common/include/sonic_common/` (and `src/common/src/`).
2. Update `gear_sonic_deploy/src/common/CMakeLists.txt` to convert
   `sonic_common` from `INTERFACE` to `STATIC` and compile the moved
   `.cpp` files there.
3. Remove the temporary
   `${CMAKE_CURRENT_SOURCE_DIR}/../g1/g1_deploy_onnx_ref/include` line
   from this file's `target_include_directories`.
4. Update G1 includes from `"file_sink.hpp"` → `<sonic_common/file_sink.hpp>`
   (and similar for the other 7 headers).
5. Have G1 link `sonic_common` instead of compiling the moved `.cpp`
   files itself. Drop them from G1's `file(GLOB_RECURSE PROJECT_SRCS ...)`.
6. Rebuild and rerun both G1 and X2 deploy targets to verify zero
   regression.

Until that PR lands, **do not add new files under
`src/common/include/sonic_common/`** — the directory is intentionally
empty so you cannot accidentally diverge a header from its G1 sibling.
New shared utilities should be authored in a feature branch that does
the move atomically.

---

## How Phase 3 (X2 deploy package) consumes this

```cmake
# gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/CMakeLists.txt
add_subdirectory(${CMAKE_SOURCE_DIR}/src/common common_build)
target_link_libraries(agi_x2_deploy_onnx_ref PRIVATE sonic_common)
```

```cpp
// X2 source files
#include "fk.hpp"            // resolved via sonic_common
#include "math_utils.hpp"
#include "state_logger.hpp"
#include "ort_session.hpp"

// Plus the X2-only header generated by the Phase 1 codegen:
#include "policy_parameters.hpp"   // resolved via x2 include path
```

Phase 3's `add_subdirectory(common ...)` call is what physically pulls
this directory into the build. Until that lands, this CMakeLists.txt is
inert — it changes nothing for the existing G1 build.
