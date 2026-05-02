# LeRobot Checkpoint Format

Layout of a LeRobot policy checkpoint (`lerobot==0.4.2`).

```
pretrained_model/
├── config.json
├── model.safetensors
├── train_config.json
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

| File | Description |
|------|-------------|
| `config.json` | Policy config (`PreTrainedConfig`): policy `type`, architecture hyperparameters, `input_features` / `output_features`. Tells `from_pretrained` which class to instantiate. |
| `model.safetensors` | Policy network weights (`state_dict` of the `nn.Module`). |
| `train_config.json` | Full training run config (`TrainPipelineConfig`): dataset, optimizer, scheduler, seed. Required for `--resume`, optional for inference. |
| `policy_preprocessor.json` | Input pipeline manifest: ordered list of processor steps applied to observations before `policy.forward()` (rename, batch, device, normalize, tokenize). |
| `policy_preprocessor_step_3_normalizer_processor.safetensors` | Per-feature observation normalization stats (mean / std / min / max) for the `normalizer_processor` step at index 3 of the preprocessor. |
| `policy_postprocessor.json` | Output pipeline manifest: steps applied to the policy's action output (unnormalize, device, optional delta-to-absolute). |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | Per-feature action stats for the `unnormalizer_processor` step at index 0 of the postprocessor. Maps normalized actions back to raw robot units. |

Stateful processor sidecars follow the pattern `{pipeline}_step_{N}_{registry_name}.safetensors`, where `N` is the step's index in the pipeline.

Inference loads in order: `config.json` → `model.safetensors` → `policy_preprocessor.json` (+ its step sidecars) → `policy_postprocessor.json` (+ its step sidecars).
