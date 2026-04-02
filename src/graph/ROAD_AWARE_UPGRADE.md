# Road-Aware HGAT-POMO Upgrade

This upgrade keeps the existing HGAT-POMO backbone and only replaces the edge
representation and route cost with a road-aware, thesis-friendly version.

## Why this scope fits a bachelor's thesis

- It keeps the stop-level complete graph, so the decoder and training loop do
  not need a full rewrite.
- It adds real-world value for last-mile delivery: road topology, signal delay,
  turning cost, and weak time dependence.
- It avoids the biggest schedule risk: no joint training with microscopic
  traffic simulators.

## Research grounding

The lightweight design below is directly inspired by the direction of recent
literature and practical open-source systems:

- Time-dependent routing on road networks matters more than Euclidean distance
  in real traffic settings:
  https://arxiv.org/abs/2004.14473
- Driver choices in last-mile delivery are affected by road and curbside
  infrastructure, not only shortest path cost:
  https://arxiv.org/abs/2301.03802
- A recent DRL line for time-dependent VRP shows that adding current time and
  time-varying cost is already meaningful without changing the whole backbone:
  https://arxiv.org/abs/2503.04085
- Open-source components that match a feasible thesis pipeline:
  - OSMnx: https://github.com/gboeing/osmnx
  - OSRM: https://github.com/Project-OSRM/osrm-backend
  - SUMO: https://github.com/eclipse-sumo/sumo

## Minimal upgrade target

Replace the original static geometric edge with:

- `road_distance`
- `travel_time_offpeak`
- `travel_time_peak`
- `signal_count`
- `turn_count`
- `left_turn_count`
- `u_turn_count`
- `one_way_factor`

The main route objective becomes:

`travel_time(current_time_bucket) + intersection_delay_penalty`

where intersection delay is approximated by:

- signal penalty = 5 s
- turn penalty = 12 s
- left turn penalty = 8 s
- U-turn penalty = 30 s

These fixed penalties are intentionally simple and align well with a thesis that
must finish on time.

## Recommended file-level integration

The following steps assume your repository still contains the same core files
visible on GitHub, such as `build_graph_pyg.py`, `td_env.py`, `main_train.py`,
and `main_eval.py`.

### `build_graph_pyg.py`

Current role, inferred from the file name:
- Build node features and pairwise graph edges for PyG.

Recommended change:
- Replace the old edge feature builder with `RoadAwareMatrices.edge_attr()`.
- Keep the graph dense at the stop level.
- Do not move to a road-segment graph for the thesis version.

Suggested edge feature order:

1. `road_distance`
2. `travel_time_offpeak`
3. `travel_time_peak`
4. `signal_count`
5. `turn_count`
6. `left_turn_count`
7. `u_turn_count`
8. `one_way_factor`

### `instance_gen.py`

Recommended change:
- Keep the original synthetic instance generator for baseline experiments.
- Add one new branch:
  - baseline branch: old Euclidean/static edge
  - road-aware branch: call `build_proxy_road_aware_matrices(...)`
- If you already have real latitude/longitude stop data, replace the proxy
  builder with `road_aware_preprocess_template.py`.

### `td_env.py`

Recommended change:
- Add a state variable for `time_bucket`, with values such as `offpeak` and
  `peak`.
- Use `travel_time_offpeak` or `travel_time_peak` depending on the active
  bucket.
- Reward should optimize route duration instead of pure geometric distance.

A simple thesis-safe update rule is:

- if total visited customers <= half of the route: `offpeak`
- else: `peak`

This is not perfect traffic modeling, but it gives you a controllable
time-dependent experiment with low implementation risk.

### `main_train.py`

Recommended change:
- Add flags like:
  - `--edge_mode static`
  - `--edge_mode road`
  - `--time_dependent true/false`
- Default the thesis experiments to:
  - baseline: `static`
  - improved: `road + time_dependent`

### `main_eval.py`

Recommended metrics:
- route travel time
- route geometric distance
- number of signalized intersections
- number of turns
- relative improvement over baseline

## Baselines and ablations

To keep the story clean, run only these three groups:

1. `Original HGAT-POMO`
   - static Euclidean edge

2. `Road-Aware HGAT-POMO`
   - real or proxy road travel time
   - no time dependence

3. `Road-Aware Time-Dependent HGAT-POMO`
   - road travel time
   - off-peak/peak two-bucket cost
   - intersection delay penalty

This experiment design answers the key thesis question:

Does better edge modeling improve last-mile routing more than only increasing
model complexity?

## Practical implementation order

1. Keep the existing training code running unchanged.
2. Add `road_aware_features.py`.
3. Let `instance_gen.py` produce the new edge matrix.
4. Feed the new edge matrix into graph construction.
5. Update `td_env.py` to use travel time as the route cost.
6. Re-run only the small/medium instance experiments first.

## What not to do now

- Do not switch to microscopic signal-phase simulation in the main thesis
  pipeline.
- Do not redesign the encoder and the environment at the same time.
- Do not add more than two time buckets unless all core experiments are already
  stable.

## A thesis-ready contribution statement

You can summarize the contribution like this:

> Existing HGAT-POMO-style methods usually optimize stop-level routes on static
> geometric or simplified pairwise costs. This thesis extends the edge
> representation to a road-aware, time-dependent form by encoding road travel
> time, turning cost, and signalized intersection delay, thereby improving the
> realism of last-mile route optimization without changing the core decoder
> architecture.
