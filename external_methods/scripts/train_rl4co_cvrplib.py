from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import dump_json, ensure_dir, repo_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL4CO on exported open CVRPLIB static data.")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="external_methods/repos/rl4co",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/train/rl4co/dataset.npz",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/val/rl4co/dataset.npz",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/test/rl4co/dataset.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_methods/results/rl4co/open_cvrplib_n30",
    )
    parser.add_argument("--model", type=str, default="am", choices=["am", "pomo"])
    parser.add_argument("--problem-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-augment", type=int, default=8)
    return parser.parse_args()


def add_repo_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def count_npz_samples(path: str | Path) -> int:
    with np.load(path) as data:
        return int(data["depot"].shape[0])


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    output_dir = ensure_dir(args.output_dir)
    add_repo_to_path(repo_dir)

    try:
        from lightning.pytorch.callbacks import ModelCheckpoint
        from rl4co.envs import CVRPEnv
        from rl4co.models.zoo import AttentionModel, POMO
        from rl4co.utils import RL4COTrainer
    except ImportError as exc:
        raise RuntimeError(
            "RL4CO dependencies are not installed in the current environment. "
            "See external_methods/docs/dependencies.md for the suggested pip commands."
        ) from exc

    train_size = count_npz_samples(args.train_dataset)
    val_size = count_npz_samples(args.val_dataset)
    test_size = count_npz_samples(args.test_dataset)

    train_path = Path(args.train_dataset).resolve()
    val_path = Path(args.val_dataset).resolve()
    test_path = Path(args.test_dataset).resolve()
    data_dir = train_path.parent

    env = CVRPEnv(
        generator_params={"num_loc": args.problem_size},
        data_dir=str(data_dir),
        train_file=train_path.name,
        val_file=val_path.name,
        test_file=test_path.name,
    )

    model_kwargs = {
        "batch_size": args.batch_size,
        "val_batch_size": args.eval_batch_size,
        "test_batch_size": args.eval_batch_size,
        "train_data_size": train_size,
        "val_data_size": val_size,
        "test_data_size": test_size,
        "optimizer_kwargs": {"lr": args.lr, "weight_decay": args.weight_decay},
    }
    if args.model == "am":
        model = AttentionModel(env, **model_kwargs)
    else:
        model = POMO(env, num_augment=args.num_augment, **model_kwargs)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
    )
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        callbacks=[checkpoint_cb],
        default_root_dir=str(output_dir),
    )

    run_config = {
        "repo_dir": repo_relative(repo_dir),
        "train_dataset": repo_relative(train_path),
        "val_dataset": repo_relative(val_path),
        "test_dataset": repo_relative(test_path),
        "output_dir": repo_relative(output_dir),
        "model": args.model,
        "problem_size": int(args.problem_size),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "accelerator": args.accelerator,
        "devices": int(args.devices),
        "num_augment": int(args.num_augment),
        "seed": int(args.seed),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    trainer.fit(model)
    test_results = trainer.test(model)

    metrics = {
        "method": f"rl4co_{args.model}",
        "problem": "static_cvrp",
        "dataset_protocol": repo_relative(Path(args.train_dataset).resolve().parents[2] / "protocol.json"),
        "run_config": run_config,
        "test_results": test_results,
        "files": {
            "run_config_json": repo_relative(output_dir / "run_config.json"),
            "checkpoints_dir": repo_relative(output_dir / "checkpoints"),
        },
    }
    dump_json(metrics, output_dir / "metrics.json")
    print(f"Finished RL4CO training. Metrics written to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
