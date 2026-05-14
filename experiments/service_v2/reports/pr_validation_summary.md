# PR Validation Summary

Branch: `feature/service-v2-joint-teacher`

PR: `https://github.com/aaaaaaaaaaa166/hgat-pomo/pull/new/feature/service-v2-joint-teacher`

## Diff Hygiene

- No new `.pt`, `.pth`, `.ckpt`, log, debug JSON, or large metrics files are included in the PR diff.
- The PR diff contains code, small Markdown reports, and `.gitignore` cleanup only.
- No token, GitHub PAT, password, private key, or local private path was found in the PR diff.
- `.gitignore` still excludes heavyweight model artifacts:
  - `*.pt`
  - `*.pth`
  - `*.ckpt`
  - `checkpoints/`
  - experiment model/log/smoke outputs
- `.gitignore` was adjusted from `models/` to `/models/` so `src/models/*.py` is not accidentally ignored.

## Existing Tracked Large Files

The repository already tracks these large baseline files; they were not added by this PR:

- `experiments/frozen_models_20260419/model_backup_ep140.pt`
- `experiments/frozen_models_20260419/model_main_ep200.pt`

No new large files were added by this branch.

## Uncommitted Local Source Changes

The following local source changes are still uncommitted and are not part of the PR:

- `src/main_eval.py`
- `src/main_train.py`
- `src/rl/pomo_rollout.py`
- `src/experiments/README.md`

Assessment:

- These look like older sequence-time-window / formal scheduler / training-log extensions.
- They are not required by the currently committed Service V2 teacher/evaluation path.
- They should not be mixed into this PR.
- Recommended handling: move to a separate branch/PR if still needed, otherwise revert after confirming no local work needs to be preserved.

The previously uncommitted `src/models/hgat_encoder.py` and `src/models/policy.py` changes were required because `ServicePolicy` depends on configurable feature dimensions. They were committed as:

- `4d0b741 fix: support configurable policy feature dims`

## Validation Commands Run

Static checks:

```powershell
python -m py_compile src/models/hgat_encoder.py src/models/policy.py src/models/service_policy.py
python -m py_compile src/experiments/eval_acceptance_insertion.py src/models/service_policy.py src/experiments/build_imitation_dataset.py src/experiments/train_service_policy.py src/experiments/eval_service_policy.py
```

Smoke checks:

```powershell
python -m src.experiments.eval_acceptance_insertion --output-dir experiments/service_v2/evaluation/git_smoke_teacher --eval-instances 1 --eval-progress-every 1 --methods raw_baseline,beam_oracle_insertion --baseline-model-path experiments/frozen_models_20260419/model_main_ep200.pt --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
python -m src.experiments.build_imitation_dataset --instances 1 --progress-every 1 --teacher-method edd_insertion --output-path experiments/service_v2/imitation/git_smoke_imitation_dataset.pt --report-path experiments/service_v2/reports/git_smoke_imitation_quality.md --dataset-path datasets/cvrplib --eval-split-file datasets/cvrplib/splits/test.txt --N 30 --K 8 --eval-seed 0
```

## PR Status

The PR is clean with respect to file hygiene, but the teacher gate does not pass at `eval_instances=10/20`. Do not train `ServicePolicy` from the current teacher.

