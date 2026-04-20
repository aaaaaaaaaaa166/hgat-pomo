from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import dump_json, ensure_dir, repo_relative


class NoBaseline:
    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        return 0, 0

    def state_dict(self):
        return {"type": "none"}

    def load_state_dict(self, state_dict):
        return None


class ExponentialBaseline(NoBaseline):
    def __init__(self, beta: float):
        self.beta = float(beta)
        self.v = None

    def eval(self, x, c):
        mean_cost = c.mean()
        if self.v is None:
            self.v = mean_cost.detach()
        else:
            self.v = (self.beta * self.v + (1.0 - self.beta) * mean_cost).detach()
        return self.v, 0

    def state_dict(self):
        return {
            "type": "exponential",
            "beta": self.beta,
            "v": None if self.v is None else float(self.v.detach().cpu().item()),
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return
        self.beta = float(state_dict.get("beta", self.beta))
        v = state_dict.get("v", None)
        self.v = None if v is None else torch.tensor(float(v))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Attention-Learn-to-Route on exported open CVRPLIB static data.")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="external_methods/repos/attention-learn-to-route",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/train/attention_learn_to_route/dataset.pkl",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/val/attention_learn_to_route/dataset.pkl",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/test/attention_learn_to_route/dataset.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_methods/results/attention_learn_to_route/open_cvrplib_n30",
    )
    parser.add_argument("--graph-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr-model", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-encode-layers", type=int, default=3)
    parser.add_argument("--tanh-clipping", type=float, default=10.0)
    parser.add_argument("--normalization", type=str, default="batch")
    parser.add_argument("--baseline", type=str, default="exponential", choices=["none", "exponential"])
    parser.add_argument("--exp-beta", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--train-size-limit", type=int, default=0, help="0 means use the full train dataset.")
    parser.add_argument("--val-size-limit", type=int, default=0, help="0 means use the full validation dataset.")
    parser.add_argument("--test-size-limit", type=int, default=0, help="0 means use the full test dataset.")
    parser.add_argument(
        "--init-model",
        type=str,
        default="",
        help="Optional checkpoint to warm-start from. Supports files produced by this wrapper or a raw state_dict.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Resume optimizer/baseline/epoch state from a checkpoint produced by this wrapper.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_pickle_samples(path: str | Path) -> int:
    with Path(path).open("rb") as f:
        return len(pickle.load(f))


def add_repo_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def make_baseline(args: argparse.Namespace):
    if args.baseline == "none":
        return NoBaseline()
    return ExponentialBaseline(beta=args.exp_beta)


def load_model_weights(model, init_model_path: str, device: torch.device) -> None:
    if not init_model_path.strip():
        return
    payload = torch.load(init_model_path.strip(), map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=True)


def load_resume_state(
    model,
    optimizer,
    baseline,
    resume_checkpoint_path: str,
    device: torch.device,
) -> int:
    if not resume_checkpoint_path.strip():
        return 0
    payload = torch.load(resume_checkpoint_path.strip(), map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    baseline.load_state_dict(payload.get("baseline", {}))
    if "rng_state" in payload:
        torch.set_rng_state(payload["rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state"):
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return int(payload["epoch"]) + 1


def save_history(history: List[Dict[str, Any]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_avg_cost",
                "train_avg_loss",
                "val_avg_cost",
                "val_std_cost",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def run_attention_batch(model, optimizer, baseline, batch, device: torch.device, max_grad_norm: float) -> tuple[float, float]:
    x, bl_val = baseline.unwrap_batch(batch)

    def move_to(var):
        if isinstance(var, dict):
            return {k: move_to(v) for k, v in var.items()}
        return var.to(device)

    x = move_to(x)
    bl_val = move_to(bl_val) if bl_val is not None else None

    cost, log_likelihood = model(x)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm if max_grad_norm > 0 else float("inf"))
    optimizer.step()
    return float(cost.mean().detach().cpu().item()), float(loss.detach().cpu().item())


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    add_repo_to_path(repo_dir)

    from nets.attention_model import AttentionModel, set_decode_type
    from train import rollout
    from utils.functions import load_problem

    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    problem = load_problem("cvrp")

    train_size_all = count_pickle_samples(args.train_dataset)
    val_size_all = count_pickle_samples(args.val_dataset)
    test_size_all = count_pickle_samples(args.test_dataset)
    train_size = train_size_all if args.train_size_limit <= 0 else min(train_size_all, args.train_size_limit)
    val_size = val_size_all if args.val_size_limit <= 0 else min(val_size_all, args.val_size_limit)
    test_size = test_size_all if args.test_size_limit <= 0 else min(test_size_all, args.test_size_limit)

    opts = SimpleNamespace(
        problem="cvrp",
        model="attention",
        graph_size=args.graph_size,
        batch_size=args.batch_size,
        epoch_size=train_size,
        val_size=val_size,
        eval_batch_size=args.eval_batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_encode_layers=args.n_encode_layers,
        tanh_clipping=args.tanh_clipping,
        normalization=args.normalization,
        checkpoint_encoder=False,
        shrink_size=None,
        max_grad_norm=args.max_grad_norm,
        log_step=50,
        no_tensorboard=True,
        no_progress_bar=args.no_progress_bar,
        device=device,
        use_cuda=device.type == "cuda",
        data_distribution=None,
        seed=args.seed,
    )

    model = AttentionModel(
        args.embedding_dim,
        args.hidden_dim,
        problem,
        n_encode_layers=args.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=args.normalization,
        tanh_clipping=args.tanh_clipping,
        checkpoint_encoder=False,
        shrink_size=None,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model)
    baseline = make_baseline(args)
    start_epoch = 0
    if args.resume_checkpoint.strip():
        start_epoch = load_resume_state(
            model=model,
            optimizer=optimizer,
            baseline=baseline,
            resume_checkpoint_path=args.resume_checkpoint,
            device=device,
        )
    else:
        load_model_weights(model, args.init_model, device)

    train_dataset = problem.make_dataset(filename=args.train_dataset, num_samples=train_size, size=args.graph_size)
    val_dataset = problem.make_dataset(filename=args.val_dataset, num_samples=val_size, size=args.graph_size)
    test_dataset = problem.make_dataset(filename=args.test_dataset, num_samples=test_size, size=args.graph_size)

    run_config = {
        "repo_dir": repo_relative(repo_dir),
        "train_dataset": repo_relative(args.train_dataset),
        "val_dataset": repo_relative(args.val_dataset),
        "test_dataset": repo_relative(args.test_dataset),
        "output_dir": repo_relative(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "graph_size": int(args.graph_size),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "epochs": int(args.epochs),
        "lr_model": float(args.lr_model),
        "baseline": args.baseline,
        "exp_beta": float(args.exp_beta),
        "model": "attention",
        "problem": "cvrp",
        "embedding_dim": int(args.embedding_dim),
        "hidden_dim": int(args.hidden_dim),
        "n_encode_layers": int(args.n_encode_layers),
        "tanh_clipping": float(args.tanh_clipping),
        "normalization": args.normalization,
        "init_model": repo_relative(args.init_model) if args.init_model.strip() else "",
        "resume_checkpoint": repo_relative(args.resume_checkpoint) if args.resume_checkpoint.strip() else "",
    }
    (output_dir / "args.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    history: List[Dict[str, Any]] = []
    best_val_cost = float("inf")
    best_epoch = None
    best_checkpoint = None

    for epoch in range(start_epoch, args.epochs):
        train_dataset_epoch = baseline.wrap_dataset(train_dataset)
        train_loader = DataLoader(
            train_dataset_epoch,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        model.train()
        set_decode_type(model, "sampling")
        step = epoch * max(1, train_size // max(1, args.batch_size))
        train_batch_costs: List[float] = []
        train_batch_losses: List[float] = []

        for batch_id, batch in enumerate(train_loader):
            batch_cost, batch_loss = run_attention_batch(
                model=model,
                optimizer=optimizer,
                baseline=baseline,
                batch=batch,
                device=device,
                max_grad_norm=args.max_grad_norm,
            )
            train_batch_costs.append(batch_cost)
            train_batch_losses.append(batch_loss)
            step += 1

        with torch.no_grad():
            set_decode_type(model, "greedy")
            model.eval()
            val_costs = rollout(model, val_dataset, opts)

        train_avg_cost = float(np.mean(train_batch_costs)) if train_batch_costs else float("nan")
        train_avg_loss = float(np.mean(train_batch_losses)) if train_batch_losses else float("nan")
        val_avg_cost = float(val_costs.mean().item())
        val_std_cost = float(val_costs.std(unbiased=False).item()) if len(val_costs) > 1 else 0.0

        history_row = {
            "epoch": epoch,
            "train_avg_cost": train_avg_cost,
            "train_avg_loss": train_avg_loss,
            "val_avg_cost": val_avg_cost,
            "val_std_cost": val_std_cost,
        }
        history.append(history_row)
        print(
            f"[epoch={epoch:03d}] train_avg_cost={train_avg_cost:.4f} "
            f"val_avg_cost={val_avg_cost:.4f} val_std={val_std_cost:.4f}"
        )

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "baseline": baseline.state_dict(),
            "epoch": epoch,
            "run_config": run_config,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        if args.checkpoint_every > 0 and ((epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1):
            torch.save(checkpoint_payload, checkpoints_dir / f"epoch-{epoch}.pt")

        if val_avg_cost < best_val_cost:
            best_val_cost = val_avg_cost
            best_epoch = epoch
            best_checkpoint = checkpoints_dir / "best-model.pt"
            torch.save(checkpoint_payload, best_checkpoint)

    if best_checkpoint is not None and best_checkpoint.exists():
        best_payload = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(best_payload["model"])

    with torch.no_grad():
        set_decode_type(model, "greedy")
        model.eval()
        test_costs = rollout(model, test_dataset, opts)

    history_path = output_dir / "history.csv"
    save_history(history, history_path)

    metrics = {
        "method": "attention_learn_to_route",
        "problem": "static_cvrp",
        "dataset_protocol": repo_relative(Path(args.train_dataset).resolve().parents[2] / "protocol.json"),
        "run_config": run_config,
        "best_epoch": best_epoch,
        "best_val_cost": best_val_cost,
        "test_avg_cost": float(test_costs.mean().item()),
        "test_std_cost": float(test_costs.std(unbiased=False).item()) if len(test_costs) > 1 else 0.0,
        "test_num_instances": int(len(test_costs)),
        "files": {
            "history_csv": repo_relative(history_path),
            "best_model_pt": None if best_checkpoint is None else repo_relative(best_checkpoint),
            "args_json": repo_relative(output_dir / "args.json"),
        },
    }
    dump_json(metrics, output_dir / "metrics.json")
    print(f"Finished Attention-Learn-to-Route training. Metrics written to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
