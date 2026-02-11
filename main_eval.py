from __future__ import annotations
import time
import numpy as np
import torch

from src.env.instance_gen import make_random_instance
from src.env.td_env import TruckDroneRendezvousEnv, EnvConfig
from src.models.policy import HGATPolicy
from src.rl.pomo_rollout import pomo_rollout


def random_rollout(env: TruckDroneRendezvousEnv, K: int = 8, max_steps: int = 256) -> np.ndarray:
    """
    Random baseline: sample feasible j and feasible k (or none).
    Return total_time for each of K trials.
    """
    times = []
    for _ in range(K):
        e = env.copy()
        obs = e.reset()
        total_time = 0.0
        done = False
        for _ in range(max_steps):
            masks = e.get_masks()
            truck_mask = masks["truck_mask"]
            js = np.where(truck_mask > 0)[0]
            j = int(np.random.choice(js))

            dmask = e.get_masks(j=j)["drone_mask"]
            ks = np.where(dmask > 0)[0]
            if len(ks) > 0 and np.random.rand() < 0.7:
                k = int(np.random.choice(ks))
            else:
                k = e.K_NONE

            obs, r, done, info = e.step((k, j))
            total_time += float(info["dt"])
            if done:
                break

        if not done:
            total_time += 1000.0  # timeout penalty
        times.append(total_time)
    return np.array(times, dtype=np.float32)


def summary(x_list):
    x = np.array(x_list, dtype=np.float32)
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def main():
    device = torch.device("cpu")
    print("Using device:", device)

    N = 30
    K = 8
    n_instances = 100
    max_steps = 5 * (N + 1)

    cfg = EnvConfig(
        vT=1.0, vD=1.5, QD=1.0, B=6.0,
        sT=0.0, sD=0.0,
        allow_wait=True,
        idle_to_next_release=True,
    )

    # Load the model, ensuring weights_only=True to avoid running arbitrary code
    policy = HGATPolicy(hidden_dim=128, heads=4, dropout=0.0, k_nn_orders=8).to(device)
    policy.load_state_dict(torch.load("policy.pt", map_location=device, weights_only=True))  # Updated line
    policy.eval()

    model_best, model_mean, model_worst = [], [], []
    rand_best, rand_mean, rand_worst = [], [], []

    best_trajs_all = []

    t0 = time.time()
    with torch.no_grad():
        for idx in range(1, n_instances + 1):
            coord, release, demand = make_random_instance(
                N=N, seed=idx, coord_scale=10.0,
                release_mode="batches", n_batches=4, max_release=10.0
            )
            env = TruckDroneRendezvousEnv(coord, release, demand, cfg=cfg, seed=idx)

            # model: K sampled trajs
            returns, _, trajs = pomo_rollout(policy, env, K=K, max_steps=max_steps, store_traj=True)
            # returns = sum reward = -total_time
            times = (-returns).numpy()  # (K,) positive total time

            model_best.append(float(times.min()))
            model_mean.append(float(times.mean()))
            model_worst.append(float(times.max()))

            best_id = int(times.argmin())
            assert trajs is not None
            best_trajs_all.append((idx, float(times[best_id]), trajs[best_id]))

            # random baseline
            rt = random_rollout(env, K=K, max_steps=max_steps)
            rand_best.append(float(rt.min()))
            rand_mean.append(float(rt.mean()))
            rand_worst.append(float(rt.max()))

            if idx % 10 == 0:
                print(f"[{idx:03d}/{n_instances}] model_best={model_best[-1]:.2f} model_mean={model_mean[-1]:.2f} | "
                      f"rand_best={rand_best[-1]:.2f} rand_mean={rand_mean[-1]:.2f}")

    print("\n=== Evaluation Summary (Total Time, smaller is better) ===")
    mb = summary(model_best); mm = summary(model_mean); mw = summary(model_worst)
    rb = summary(rand_best);  rm = summary(rand_mean);  rw = summary(rand_worst)

    print(f"MODEL best(K): mean={mb[0]:.3f}, std={mb[1]:.3f}, best(min)={mb[2]:.3f}, worst(max)={mb[3]:.3f}")
    print(f"MODEL mean(K): mean={mm[0]:.3f}, std={mm[1]:.3f}, best(min)={mm[2]:.3f}, worst(max)={mm[3]:.3f}")
    print(f"MODEL worst(K): mean={mw[0]:.3f}, std={mw[1]:.3f}, best(min)={mw[2]:.3f}, worst(max)={mw[3]:.3f}")
    print(f"RAND  best(K): mean={rb[0]:.3f}, std={rb[1]:.3f}, best(min)={rb[2]:.3f}, worst(max)={rb[3]:.3f}")
    print(f"RAND  mean(K): mean={rm[0]:.3f}, std={rm[1]:.3f}, best(min)={rm[2]:.3f}, worst(max)={rm[3]:.3f}")
    print(f"RAND  worst(K): mean={rw[0]:.3f}, std={rw[1]:.3f}, best(min)={rw[2]:.3f}, worst(max)={rw[3]:.3f}")

    # save best trajectories
    with open(f"eval_trajs_N{N}.txt", "w", encoding="utf-8") as f:
        for inst_id, tbest, traj in best_trajs_all:
            f.write(f"instance={inst_id} best_time={tbest:.6f}\n")
            for step_id, item in enumerate(traj):
                action = item["action"]
                info = item["info"]
                if action == ("TIMEOUT",):
                    f.write(f"  step={step_id:03d} TIMEOUT\n")
                    continue
                k, j = action
                f.write(
                    f"  step={step_id:03d} i={info.get('i')} j={j} k={k} "
                    f"dt={info.get('dt', 0.0):.6f} truck={info.get('truck_time', 0.0):.6f} "
                    f"drone={info.get('drone_time', 0.0):.6f}\n"
                )
            f.write("\n")

    print(f"\nSaved best trajectories to eval_trajs_N{N}.txt")
    print(f"Total eval time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
