# LeRobot Model Checkpoint Format

This page documents the on-disk layout of a LeRobot policy checkpoint as produced by `lerobot==0.4.2` (the version pinned in `packages/simulator/pyproject.toml`). It is aimed at engineers who need to load, audit, or hand-modify a trained checkpoint without going through `from_pretrained`.

## Where these files live

`lerobot.utils.train_utils.save_checkpoint` writes one directory per saved step:

```
<output_dir>/checkpoints/<step>/
├── pretrained_model/        # everything documented below
└── training_state/          # optimizer / scheduler / RNG (resume only — not used at inference)
```

Only `pretrained_model/` is needed for inference and for `--policy.path=` / `lerobot-eval`. A typical layout:

```
pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json
```

The exact set of `*_step_<N>_*.safetensors` files depends on the preprocessor / postprocessor pipeline configured for the policy — only stateful steps emit a `.safetensors` sidecar. The example above corresponds to the default normalize-on-input / unnormalize-on-output pipeline.

## File reference

### `config.json`

Serialized `PreTrainedConfig` (a.k.a. the policy config). Written by `policy.config.save_pretrained(...)` using the `huggingface_hub` `CONFIG_NAME` constant.

Contents:
- `type` — policy class discriminator (e.g. `act`, `diffusion`, `pi0`, `smolvla`). Drives which subclass `PreTrainedPolicy.from_pretrained` instantiates.
- Architecture hyperparameters — chunk size, hidden dims, vision backbone, transformer depth, etc.
- `input_features` / `output_features` — declared shapes and dtypes of the observation and action tensors the policy expects. Used by the processor pipeline to wire normalization stats and by callers to build a matching `LeRobotDataset` schema.
- `device`, `use_amp`, and other runtime flags consumed at inference.

Role in loading: read first by `PreTrainedPolicy.from_pretrained` to pick the policy class and rebuild the empty model graph before weights are loaded.

### `model.safetensors`

Policy network weights in `safetensors` format, written by the `huggingface_hub` `PyTorchModelHubMixin` via `policy.save_pretrained(...)`. Contains the full `state_dict` of the `nn.Module` (vision encoder, transformer, action head, …). For PEFT / LoRA runs only the adapter weights land here and the base model is referenced from `config.json`.

Role in loading: loaded into the model graph constructed from `config.json`. This is the only file holding learned parameters.

### `train_config.json`

Serialized `TrainPipelineConfig` (`lerobot.configs.train`) — the full training run config: dataset repo id, optimizer + scheduler, batch size, num steps, seed, wandb settings, evaluation cadence, and the embedded policy / env configs.

Role:
- Inference / eval: optional. Useful for reproducing the dataset/env that the policy was trained against.
- Resume: required. `lerobot-train --config_path=<checkpoint>/pretrained_model/train_config.json` rebuilds the full training pipeline.
- Migration tooling: `lerobot.processor.migrate_policy_normalization` reads it to recover dataset stats when upgrading older checkpoints to the processor format.

### `policy_preprocessor.json`

Pipeline manifest for the **input** `DataProcessorPipeline` (registered name `policy_preprocessor`). Written by `preprocessor.save_pretrained(...)` (`lerobot/processor/pipeline.py:_save_pretrained`).

Schema:
```jsonc
{
  "name": "policy_preprocessor",
  "steps": [
    {
      "registry_name": "<step name>",   // or "class": "<dotted import path>"
      "config": { ... },                // step.get_config() output
      "state_file": "<filename>"        // present only for stateful steps
    },
    ...
  ]
}
```

Typical step order for a vision-language policy:
1. `rename_observations_processor` — map dataset keys to model-expected keys.
2. `add_batch_dimension_processor` — promote single transitions to a batch.
3. `device_processor` — move tensors to the policy device / dtype.
4. `normalizer_processor` — z-score / min-max scale observations using dataset stats. **Stateful** → emits step 3 `.safetensors`.
5. `tokenizer_processor` (VLA only) — tokenize the language instruction.

Role in inference: `DataProcessorPipeline.from_pretrained(..., config_filename="policy_preprocessor.json")` rebuilds the pipeline. Each call site composes `policy(preprocessor(transition))`.

### `policy_preprocessor_step_3_normalizer_processor.safetensors`

State for the `NormalizerProcessorStep` at index 3 of the preprocessor pipeline. Filename pattern is fixed by `_save_pretrained`:

```
{sanitized_pipeline_name}_step_{step_index}_{registry_name}.safetensors
```

So `step_3` here means "the fourth element of `policy_preprocessor.steps`", and the suffix matches the `@ProcessorStepRegistry.register(name="normalizer_processor")` decorator on `NormalizerProcessorStep`.

Tensor contents: per-feature normalization statistics computed from the training dataset — `mean`, `std`, `min`, `max`, `count`, organized by feature key (e.g. `observation.state`, `observation.images.front`). Used to convert raw observations into the zero-mean / unit-variance space the network was trained on.

Role: loaded via `safetensors.load_file` during `from_pretrained` and copied into the `NormalizerProcessorStep` buffers. Without this file the normalizer would have zero stats and inference would silently produce garbage.

### `policy_postprocessor.json`

Pipeline manifest for the **output** `DataProcessorPipeline` (registered name `policy_postprocessor`). Same schema as `policy_preprocessor.json`. It runs on whatever the policy returns from `forward()` / `select_action()`.

Typical step order:
0. `unnormalizer_processor` — invert the action normalization so the policy emits actions in raw robot units. **Stateful** → emits step 0 `.safetensors`.
1. `device_processor` — move actions back to CPU / target dtype.
2. `delta_action_processor` (optional) — convert delta actions to absolute joint targets.

Role in inference: loaded with `config_filename="policy_postprocessor.json"`. The runtime contract is `action = postprocessor(policy(preprocessor(obs)))`.

### `policy_postprocessor_step_0_unnormalizer_processor.safetensors`

State for the `UnnormalizerProcessorStep` at index 0 of the postprocessor pipeline. Same naming pattern as the preprocessor sidecars; the `unnormalizer_processor` suffix comes from `@ProcessorStepRegistry.register(name="unnormalizer_processor")` on `UnnormalizerProcessorStep`.

Tensor contents: the same per-feature stats schema as the normalizer file, but keyed on action features (e.g. `action`). The unnormalizer applies the inverse of the training-time action normalization so that downstream consumers (Isaac Lab action manager, real robot driver) receive actions in joint / Cartesian units.

Role: loaded into the `UnnormalizerProcessorStep` buffers. Mismatched stats here are the most common cause of "policy outputs look fine but the robot moves to the wrong pose" bugs after a checkpoint move.

## End-to-end load order

`PreTrainedPolicy.from_pretrained(<checkpoint>/pretrained_model)` and the matching `DataProcessorPipeline.from_pretrained` calls perform, in order:

1. Read `config.json` → instantiate the empty policy.
2. Load `model.safetensors` → populate weights.
3. Read `policy_preprocessor.json` → rebuild input pipeline; for every step with a `state_file`, load the matching `*_step_<N>_*.safetensors` into its buffers.
4. Read `policy_postprocessor.json` → same, for the output pipeline.
5. (Optional) read `train_config.json` if the caller needs dataset / env metadata or is resuming training.

## Notes

- Pre-0.3 LeRobot checkpoints predate the `policy_preprocessor` / `policy_postprocessor` split — normalization stats lived inside `model.safetensors` under `normalize_inputs.*`. Run `python -m lerobot.processor.migrate_policy_normalization --pretrained_path <ckpt>` to upgrade.
- The `step_<N>_<registry_name>` filename embeds the pipeline ordering at save time. Reordering the steps changes the filenames, so do not rename these files by hand.
- `training_state/` (optimizer + scheduler + RNG) lives next to `pretrained_model/` and is only consumed by `lerobot-train --resume`. It is never required for inference and is safe to drop when shipping a checkpoint.
