# External Method Dependencies

## Current repo environment

The nested repo virtualenv at `.venv/.venv` currently contains the main HGAT-POMO stack, but does not yet include the full RL4CO dependency chain.

## Practical split

- `train_pomo_cvrplib.py`
  - Expected to run with the current base PyTorch environment.

- `train_attention_cvrplib.py`
  - Uses a lightweight local wrapper and avoids TensorBoard by default.
  - If you later want the original upstream training script, install `tensorboard_logger`.

- `train_rl4co_cvrplib.py`
  - Needs the RL4CO stack first.

## Suggested install commands

If you want to enable RL4CO inside the current nested venv:

```powershell
& .\.venv\.venv\Scripts\pip.exe install lightning torchrl tensordict hydra-core omegaconf
```

If you want upstream Attention-Learn-to-Route conveniences:

```powershell
& .\.venv\.venv\Scripts\pip.exe install tensorboard_logger scipy
```

These are intentionally documented here instead of being auto-installed so the workspace stays auditable.
