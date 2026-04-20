from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from common import dump_json, ensure_dir, repo_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train POMO on exported open CVRPLIB static data.")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="external_methods/repos/POMO",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/train/pomo/dataset.pt",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/val/pomo/dataset.pt",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="external_methods/data/open_cvrplib_n30/test/pomo/dataset.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_methods/results/pomo/open_cvrplib_n30",
    )
    parser.add_argument("--problem-size", type=int, default=30)
    parser.add_argument("--pomo-size", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--encoder-layer-num", type=int, default=6)
    parser.add_argument("--qkv-dim", type=int, default=16)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--ff-hidden-dim", type=int, default=512)
    parser.add_argument("--logit-clipping", type=float, default=10.0)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--train-episodes", type=int, default=0, help="0 means use the full saved train set each epoch.")
    parser.add_argument("--val-episodes", type=int, default=0, help="0 means use the full saved val set.")
    parser.add_argument("--test-episodes", type=int, default=0, help="0 means use the full saved test set.")
    parser.add_argument("--test-aug-factor", type=int, default=8)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--init-model",
        type=str,
        default="",
        help="Optional checkpoint to warm-start from. Supports this wrapper's checkpoint payloads or raw state_dict files.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Resume optimizer/history/epoch state from a checkpoint produced by this wrapper.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_repo_to_path(repo_dir: Path) -> None:
    base_dir = (repo_dir / "NEW_py_ver").resolve()
    cvrp_dir = (base_dir / "CVRP").resolve()
    pomo_dir = (base_dir / "CVRP" / "POMO").resolve()
    for path in [str(pomo_dir), str(cvrp_dir), str(base_dir)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def configure_device(no_cuda: bool) -> torch.device:
    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device


def load_dataset(path: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    return {
        "depot_xy": payload["depot_xy"].to(device),
        "node_xy": payload["node_xy"].to(device),
        "node_demand": payload["node_demand"].to(device),
    }


def dataset_size(payload: Dict[str, torch.Tensor]) -> int:
    return int(payload["depot_xy"].shape[0])


def load_model_weights(model, init_model_path: str, device: torch.device) -> None:
    if not init_model_path.strip():
        return
    payload = torch.load(init_model_path.strip(), map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=True)


def load_resume_state(model, optimizer, resume_checkpoint_path: str, device: torch.device) -> tuple[int, list]:
    if not resume_checkpoint_path.strip():
        return 1, []
    payload = torch.load(resume_checkpoint_path.strip(), map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    history = list(payload.get("history", []))
    start_epoch = int(payload["epoch"]) + 1
    return start_epoch, history


def load_saved_problems_into_env(env, payload: Dict[str, torch.Tensor], shuffle: bool = False, seed: int = 0, limit: int = 0) -> int:
    if shuffle:
        generator = torch.Generator(device=payload["depot_xy"].device)
        generator.manual_seed(seed)
        perm = torch.randperm(dataset_size(payload), generator=generator, device=payload["depot_xy"].device)
        depot_xy = payload["depot_xy"][perm]
        node_xy = payload["node_xy"][perm]
        node_demand = payload["node_demand"][perm]
    else:
        depot_xy = payload["depot_xy"]
        node_xy = payload["node_xy"]
        node_demand = payload["node_demand"]

    if limit > 0:
        depot_xy = depot_xy[:limit]
        node_xy = node_xy[:limit]
        node_demand = node_demand[:limit]

    env.FLAG__use_saved_problems = True
    env.saved_depot_xy = depot_xy
    env.saved_node_xy = node_xy
    env.saved_node_demand = node_demand
    env.saved_index = 0
    return int(depot_xy.shape[0])


def train_one_batch(model, env, optimizer, batch_size: int) -> Tuple[float, float]:
    model.train()
    env.load_problems(batch_size)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
    state, reward, done = env.pre_step()

    while not done:
        selected, prob = model(state)
        state, reward, done = env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

    advantage = reward - reward.float().mean(dim=1, keepdims=True)
    log_prob = prob_list.log().sum(dim=2)
    loss = -advantage * log_prob
    loss_mean = loss.mean()

    max_pomo_reward, _ = reward.max(dim=1)
    score_mean = -max_pomo_reward.float().mean()

    optimizer.zero_grad()
    loss_mean.backward()
    optimizer.step()
    return float(score_mean.item()), float(loss_mean.item())


def evaluate_dataset(model, env, payload: Dict[str, torch.Tensor], batch_size: int, aug_factor: int = 1, limit: int = 0) -> Dict[str, float]:
    num_episodes = load_saved_problems_into_env(env, payload, shuffle=False, limit=limit)
    score_values: List[float] = []
    aug_score_values: List[float] = []

    processed = 0
    model.eval()
    with torch.no_grad():
        while processed < num_episodes:
            cur_batch = min(batch_size, num_episodes - processed)
            env.load_problems(cur_batch, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)

            aug_reward = reward.reshape(aug_factor, cur_batch, env.pomo_size)
            max_pomo_reward, _ = aug_reward.max(dim=2)
            no_aug_score = -max_pomo_reward[0, :].float()
            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
            aug_score = -max_aug_pomo_reward.float()

            score_values.extend(no_aug_score.detach().cpu().tolist())
            aug_score_values.extend(aug_score.detach().cpu().tolist())
            processed += cur_batch

    score_arr = np.asarray(score_values, dtype=np.float32)
    aug_arr = np.asarray(aug_score_values, dtype=np.float32)
    return {
        "num_instances": int(len(score_arr)),
        "no_aug_avg_cost": float(score_arr.mean()) if len(score_arr) else math.nan,
        "no_aug_std_cost": float(score_arr.std()) if len(score_arr) else math.nan,
        "aug_avg_cost": float(aug_arr.mean()) if len(aug_arr) else math.nan,
        "aug_std_cost": float(aug_arr.std()) if len(aug_arr) else math.nan,
    }


def save_history(history: List[Dict[str, Any]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_score",
                "train_loss",
                "val_no_aug_avg_cost",
                "val_aug_avg_cost",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    output_dir = ensure_dir(args.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    add_repo_to_path(repo_dir)
    device = configure_device(args.no_cuda)
    set_seed(args.seed)

    from CVRPEnv import CVRPEnv
    from CVRPModel import CVRPModel

    train_payload = load_dataset(args.train_dataset, device)
    val_payload = load_dataset(args.val_dataset, device)
    test_payload = load_dataset(args.test_dataset, device)

    train_size = dataset_size(train_payload)
    val_size = dataset_size(val_payload)
    test_size = dataset_size(test_payload)
    train_episodes = train_size if args.train_episodes <= 0 else min(args.train_episodes, train_size)
    val_episodes = val_size if args.val_episodes <= 0 else min(args.val_episodes, val_size)
    test_episodes = test_size if args.test_episodes <= 0 else min(args.test_episodes, test_size)

    env_params = {
        "problem_size": args.problem_size,
        "pomo_size": args.pomo_size,
    }
    model_params = {
        "embedding_dim": args.embedding_dim,
        "sqrt_embedding_dim": args.embedding_dim ** 0.5,
        "encoder_layer_num": args.encoder_layer_num,
        "qkv_dim": args.qkv_dim,
        "head_num": args.head_num,
        "logit_clipping": args.logit_clipping,
        "ff_hidden_dim": args.ff_hidden_dim,
        "eval_type": "argmax",
    }
    env = CVRPEnv(**env_params)
    model = CVRPModel(**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    history: List[Dict[str, Any]] = []
    if args.resume_checkpoint.strip():
        start_epoch, history = load_resume_state(
            model=model,
            optimizer=optimizer,
            resume_checkpoint_path=args.resume_checkpoint,
            device=device,
        )
    else:
        load_model_weights(model, args.init_model, device)

    run_config = {
        "repo_dir": repo_relative(repo_dir),
        "train_dataset": repo_relative(args.train_dataset),
        "val_dataset": repo_relative(args.val_dataset),
        "test_dataset": repo_relative(args.test_dataset),
        "output_dir": repo_relative(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "problem_size": int(args.problem_size),
        "pomo_size": int(args.pomo_size),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "train_episodes": int(train_episodes),
        "val_episodes": int(val_episodes),
        "test_episodes": int(test_episodes),
        "train_batch_size": int(args.train_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "test_aug_factor": int(args.test_aug_factor),
        "init_model": repo_relative(args.init_model) if args.init_model.strip() else "",
        "resume_checkpoint": repo_relative(args.resume_checkpoint) if args.resume_checkpoint.strip() else "",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    best_val_cost = float("inf")
    best_epoch = None
    best_checkpoint = None
    if history:
        valid_costs = [row["val_no_aug_avg_cost"] for row in history if "val_no_aug_avg_cost" in row]
        if valid_costs:
            best_val_cost = float(min(valid_costs))
        if history:
            best_epoch = int(min(history, key=lambda row: row.get("val_no_aug_avg_cost", float("inf")))["epoch"])

    for epoch in range(start_epoch, args.epochs + 1):
        available = load_saved_problems_into_env(
            env,
            train_payload,
            shuffle=True,
            seed=args.seed + epoch,
            limit=train_episodes,
        )
        episode = 0
        train_score_sum = 0.0
        train_loss_sum = 0.0

        while episode < available:
            batch_size = min(args.train_batch_size, available - episode)
            score, loss = train_one_batch(model, env, optimizer, batch_size)
            train_score_sum += score * batch_size
            train_loss_sum += loss * batch_size
            episode += batch_size

        train_score = train_score_sum / max(1, available)
        train_loss = train_loss_sum / max(1, available)
        val_metrics = evaluate_dataset(
            model,
            env,
            val_payload,
            batch_size=args.eval_batch_size,
            aug_factor=1,
            limit=val_episodes,
        )

        history_row = {
            "epoch": epoch,
            "train_score": train_score,
            "train_loss": train_loss,
            "val_no_aug_avg_cost": val_metrics["no_aug_avg_cost"],
            "val_aug_avg_cost": val_metrics["aug_avg_cost"],
        }
        history.append(history_row)

        print(
            f"[epoch={epoch:03d}] train_score={train_score:.4f} train_loss={train_loss:.4f} "
            f"val_cost={val_metrics['no_aug_avg_cost']:.4f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_config": run_config,
            "history": history,
        }
        if args.checkpoint_every > 0 and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
            torch.save(checkpoint_payload, checkpoints_dir / f"checkpoint-{epoch}.pt")

        if val_metrics["no_aug_avg_cost"] < best_val_cost:
            best_val_cost = val_metrics["no_aug_avg_cost"]
            best_epoch = epoch
            best_checkpoint = checkpoints_dir / "best-model.pt"
            torch.save(checkpoint_payload, best_checkpoint)

    if best_checkpoint is not None and best_checkpoint.exists():
        payload = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])

    test_metrics = evaluate_dataset(
        model,
        env,
        test_payload,
        batch_size=args.eval_batch_size,
        aug_factor=max(1, args.test_aug_factor),
        limit=test_episodes,
    )

    history_path = output_dir / "history.csv"
    save_history(history, history_path)

    metrics = {
        "method": "pomo",
        "problem": "static_cvrp",
        "dataset_protocol": repo_relative(Path(args.train_dataset).resolve().parents[2] / "protocol.json"),
        "run_config": run_config,
        "best_epoch": best_epoch,
        "best_val_cost": best_val_cost,
        "test_metrics": test_metrics,
        "files": {
            "history_csv": repo_relative(history_path),
            "best_model_pt": None if best_checkpoint is None else repo_relative(best_checkpoint),
            "run_config_json": repo_relative(output_dir / "run_config.json"),
        },
    }
    dump_json(metrics, output_dir / "metrics.json")
    print(f"Finished POMO training. Metrics written to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
