# Exporting a Self-Implemented Env Config as a Standalone File

> Companion to [`isaaclab_leisaac_tutorial.md`](./isaaclab_leisaac_tutorial.md). Read that first — this page assumes you already understand `SingleArmFrankaTaskEnvCfg`, the manager-based config tree, and how `gym.register` wires `env_cfg_entry_point` to a config class.

This page covers one rule:

> **If you self-implement an environment configuration, you _must_ export it as a standalone configuration file.**

"Self-implement" means: you wrote a new `ManagerBasedRLEnvCfg` subclass (or any extension of `SingleArmFrankaTaskEnvCfg`) outside the in-tree task templates — for example, you tweaked the cup-stacking config inline in a notebook, defined a new env in a script, or composed one ad-hoc in a REPL.

In that case the config object that lives in memory at training time is **not** something a downstream rollout/training run can pick up automatically. You have to commit it to disk as a standalone file.

---

## 1. Why standalone export is required

### 1.1 Decoupling from in-tree configs

The in-tree configs under `packages/simulator/src/simulator/tasks/` are **shared, versioned, and centrally maintained**. Mutating them locally (e.g. monkey-patching fields on `CupStackingEnvCfg` from a notebook) creates a config that:

- has no file backing it,
- diverges silently from `main`, and
- only exists for the lifetime of the Python process.

Exporting your env config to its own file decouples it from the shared templates. You can iterate on your task without touching files other contributors depend on, and reviewers can read the exact config you trained against.

### 1.2 Reproducibility

A run is reproducible only if the env config can be re-instantiated bit-for-bit later. In-memory tweaks (monkey-patching `init_state`, mutating `terminations`, swapping a camera resolution) leave no audit trail. A standalone file:

- pins every field that differs from the parent template,
- is hashable / diff-able / git-trackable,
- can be re-imported by anyone who clones the repo,
- and survives Python session restarts.

If the config can't be re-imported from a file path, the resulting demos and policies cannot be considered reproducible — even if your run logs look fine.

### 1.3 Rollout / training pickup

Both datagen and downstream training/rollout discover envs through `gym.register(... env_cfg_entry_point=...)`. The entry point is a **`module:Class` string**. That string only resolves if the module exists as an importable file on `sys.path`.

In other words: if your config isn't a standalone file with a `gym.register` block pointing at it, `parse_env_cfg(args_cli.task, ...)` in `scripts/datagen/generate.py` (and any rollout/training script that follows the same contract) cannot construct your env. Standalone export is what makes `--task LeIsaac-HCIS-<MyTask>-...-v0` work end-to-end.

---

## 2. How to export

Treat the in-tree `cup_stacking` task as the canonical layout to mirror. The export is two files plus one registration call.

### 2.1 File location

Place the exported config under:

```
packages/simulator/src/simulator/tasks/<my_task>/
    __init__.py             # gym.register block
    <my_task>_env_cfg.py    # the standalone env config
    mdp/                    # optional — task-specific termination/reward terms
        __init__.py
```

Rules:

- One directory per task. Do not put multiple unrelated env configs in the same module.
- Keep the directory name `snake_case`, matching the task slug.
- Anything imported from this directory must be reachable from `simulator.tasks.<my_task>` — no sibling `sys.path` hacks.
- Do **not** export the file outside `packages/simulator/src/simulator/tasks/`. Other locations are not auto-discovered when `import simulator.tasks` runs.

### 2.2 Naming convention

| Artifact | Pattern | Example |
|----------|---------|---------|
| Directory | `<task_slug>` (snake_case) | `cup_stacking` |
| Config module | `<task_slug>_env_cfg.py` | `cup_stacking_env_cfg.py` |
| Config class | `<TaskName>EnvCfg` (PascalCase) | `CupStackingEnvCfg` |
| Scene class | `<TaskName>SceneCfg` | `CupStackingSceneCfg` |
| Gym id | `LeIsaac-HCIS-<TaskName>-<RobotKind>-v<N>` | `LeIsaac-HCIS-CupStacking-SingleArm-v0` |

Bump `vN` whenever you make a backwards-incompatible change to the config so that previously recorded demos stay associated with the env they were collected against.

### 2.3 Expected schema

The exported `<my_task>_env_cfg.py` must:

1. Subclass `SingleArmFrankaTaskEnvCfg` (or another in-tree base) — never a bare `ManagerBasedRLEnvCfg`. The template wires sensors, observations, teleop, and the LeRobot frame builder you almost certainly want.
2. Use `@configclass` dataclasses for every nested config (scene, observations, terminations, …). Plain Python classes will not be picked up by Isaac Lab's manager system.
3. Set `task_description: str` to a short natural-language string. It is baked into every LeRobot frame's `"task"` field and used as conditioning for VLA policies.
4. Expose all task-specific tweaks as **fields** of the config class, not as mutations applied after construction. Field defaults are part of the file; post-hoc mutations are not.
5. Implement `__post_init__(self)` for anything that must run after the parent populates defaults — viewer pose, robot home pose, `parse_usd_and_create_subassets`, the optional `object_poses_path` block. Always call `super().__post_init__()` first.
6. Use `MISSING` for any field that is intentionally filled in at runtime (e.g. teleop-device-dependent action terms). Do **not** leave fields as `None` to mean "fill in later".

Skeleton:

```python
# packages/simulator/src/simulator/tasks/<my_task>/<my_task>_env_cfg.py
from isaaclab.utils import configclass
from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaObservationsCfg,
)


@configclass
class MyTaskSceneCfg(SingleArmFrankaTaskSceneCfg):
    # scene + per-object RigidObjectCfg fields
    ...


@configclass
class MyTaskEnvCfg(SingleArmFrankaTaskEnvCfg):
    scene: MyTaskSceneCfg = MyTaskSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    task_description: str = "<short natural-language goal>"

    def __post_init__(self) -> None:
        super().__post_init__()
        # viewer, robot init_state, parse_usd_and_create_subassets, object_poses block
        ...
```

And the registration:

```python
# packages/simulator/src/simulator/tasks/<my_task>/__init__.py
import gymnasium as gym


gym.register(
    id="LeIsaac-HCIS-<MyTask>-SingleArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.<my_task>_env_cfg:MyTaskEnvCfg",
    },
)
```

`entry_point` is **always** `isaaclab.envs:ManagerBasedRLEnv` for tasks that subclass the manager-based template — only the `env_cfg_entry_point` string changes.

### 2.4 Verifying the export

After exporting, sanity-check that the file is what gets loaded:

```bash
python -c "
import simulator.tasks  # registers the env
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg('LeIsaac-HCIS-<MyTask>-SingleArm-v0', num_envs=1)
print(type(cfg).__module__, type(cfg).__qualname__)
print(cfg.task_description)
"
```

The printed module path must point at your standalone file (e.g. `simulator.tasks.<my_task>.<my_task>_env_cfg`). If it points anywhere else — a notebook, `__main__`, an in-tree template — the export is not actually being picked up.

Then run the smoke test from `isaaclab_leisaac_tutorial.md` §5 with your new task id.

---

## 3. Checklist

Before you call your env config "exported":

- [ ] Lives at `packages/simulator/src/simulator/tasks/<my_task>/<my_task>_env_cfg.py`.
- [ ] Class name follows `<TaskName>EnvCfg`.
- [ ] Subclasses an in-tree template (e.g. `SingleArmFrankaTaskEnvCfg`).
- [ ] All task-specific values are **fields** with defaults, not post-hoc mutations.
- [ ] `__post_init__` calls `super().__post_init__()` first.
- [ ] `task_description` is set.
- [ ] Sibling `__init__.py` calls `gym.register(...)` with `env_cfg_entry_point` pointing at the standalone file.
- [ ] `parse_env_cfg(<task id>, num_envs=1)` returns an instance whose module path is the standalone file.
- [ ] File is committed to git on a feature branch off `main` — not living only in a notebook, scratch script, or local-only path.

---

## 4. External / private rollout (file outside the repo tree)

For private leaderboard tasks, secret eval scenes, and per-student variants
the env config must stay **out of the public `aicapstone` tree**.
`scripts/rollout.py` accepts `--task` in three forms, resolved by
`simulator.tasks.external.resolve_task` (see AUT-80):

| Form | Example |
|---|---|
| Registered gym id | `--task LeIsaac-HCIS-CupStacking-SingleArm-v0` |
| Path to `.py` file | `--task /tmp/private_smoke/private_cup_stacking_smoke.py` |
| `module:Class` ref | `--task my_pkg.tasks.eval:MyEvalCfg` |

The external file must subclass an in-tree template, define a unique
`TASK_ID`, and call `gym.register` at import time:

```python
# /tmp/private_smoke/private_cup_stacking_smoke.py
import gymnasium as gym
from isaaclab.utils import configclass
from simulator.tasks.cup_stacking.cup_stacking_env_cfg import CupStackingEnvCfg


@configclass
class PrivateCupStackingSmokeEnvCfg(CupStackingEnvCfg):
    task_description: str = "AUT-85 private smoke."

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (0.85, 0.90, 0.70)


TASK_ID = "Private-Smoke-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:PrivateCupStackingSmokeEnvCfg"},
)
```

### 4.1 Reproducing the AUT-85 smoke locally

From the host:

```bash
make launch-isaaclab
```

Inside the container shell:

```bash
# Stage the private file *outside* /workspace/aicapstone to prove there is
# no in-tree side effect.
mkdir -p /tmp/private_smoke
cat > /tmp/private_smoke/private_cup_stacking_smoke.py <<'PY'
# (paste the snippet from §4 above)
PY

# External .py path
python scripts/rollout.py \
    --task /tmp/private_smoke/private_cup_stacking_smoke.py \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=tiny-diff \
    --policy_action_horizon=1 \
    --device=cuda \
    --enable_cameras

# Regression — same script with the in-tree gym id
python scripts/rollout.py \
    --task LeIsaac-HCIS-CupStacking-SingleArm-v0 \
    --policy_type=lerobot-diffusion \
    --policy_checkpoint_path=tiny-diff \
    --policy_action_horizon=1 \
    --device=cuda \
    --enable_cameras
```

The external invocation logs `Parsing configuration from:
_aicapstone_external_task_<hex>:PrivateCupStackingSmokeEnvCfg`; the gym-id
invocation logs `Parsing configuration from:
simulator.tasks.cup_stacking.cup_stacking_env_cfg:CupStackingEnvCfg`. Beyond
that loader-prefix difference the two runs follow identical code paths.

### 4.2 Constraints

- The external file must not be imported before
  `simulation_app = AppLauncher(...).app` finishes booting — it pulls in
  `isaaclab.*`. `scripts/rollout.py` already orders the import correctly.
- The synthetic module name `_aicapstone_external_task_<hex>` is inserted
  into `sys.modules` before `exec_module`, so `parse_env_cfg` can resolve
  `f"{__name__}:<Class>"` back through `importlib.import_module`.
- Repeated `resolve_task` calls on the same absolute path within one process
  short-circuit through `_FILE_LOAD_CACHE` and skip re-registering.
- Asset paths inside the external file should use `simulator.ASSETS_ROOT`
  for shared USDs and absolute paths for any private USDs.

---

## See also

- [`isaaclab_leisaac_tutorial.md`](./isaaclab_leisaac_tutorial.md) — full walkthrough of the task config tree, the `SingleArmFrankaTaskEnvCfg` template, the cup-stacking reference task, gym registration, and common pitfalls.
- `packages/simulator/src/simulator/tasks/cup_stacking/` — the canonical example of a fully-exported standalone task config.
- `packages/simulator/src/simulator/tasks/template/single_arm_franka_cfg.py` — the base class your standalone config should subclass.
- `packages/simulator/src/simulator/tasks/external.py` — `resolve_task` loader for the three `--task` forms.
