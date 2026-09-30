<p align="center">
  <img src="https://datasciencefestival.com/wp-content/uploads/2023/09/google-deepmind-logo.webp" alt="Google DeepMind" width="320">
</p>

# Long-horizon forecasting of ten black-box simulators under a small experiment budget

Technical notes on my entry to the Google DeepMind GroundTruth challenge (Toronto, September 2026). I joined with about 40 hours left. AI coding agents wrote and ran most of the code; I chose directions, read the leaderboard feedback and decided what to upload. Model choices were validated against Public leaderboard scores wherever possible, because local cross-validation proved unreliable (see *Process*).

## Problem

Ten hidden simulators (epidemic, market, traffic, power grid, supply chain, wildlife, reservoir, ad auction, social contagion, hospital queue), each with 2–6 bounded controls and 2–4 noisy observables. Per system: 2,000 simulator steps for research, then a `predict(initial, interventions, context)` submission that must return **4,000 open-loop steps** from a noisy initial reading and the full action schedule, NumPy/SciPy only, 40 hidden episodes, 1,200 s total.

Score per (observable, tick): `1 / (1 + |e| / σ)`, with σ hidden. Episodes are split evenly into sustained operation, action order, recovery history and composition. Each system has three candidate memory mechanisms, exactly two of which are active.

The main difficulty is generalization in **horizon** (research runs of about 250–510 steps against 4,000-step scoring) and in **schedule** (long constant holds and pulse-recovery sequences that a random research policy rarely produces).

## Results (Public leaderboard, mean over 10 systems)

| Stage | Score |
|---|---|
| Organizers' one-step linear starter, one system submitted | 0.039 |
| Ridge/starter models, all ten systems | 0.496 |
| GRU state-space models | 0.686 |
| + reservoir inflow model, social-contagion simulator blend | 0.73 (rank 22 of 125 on the published board) |
| Final submission (best proven model per system) | 0.757 expected from Public scores; Final not yet published |

Final per-system Public scores: ad auction 0.867, traffic 0.821, wildlife 0.799, reservoir 0.788, power grid 0.746, supply chain 0.745, epidemic 0.743, hospital queue 0.711, market 0.701, social contagion 0.655. Final uses different hidden episodes, so expect a shift of a few thousandths to about 0.01.

## Methods

**Evaluation protocol.** Five research runs per system (three random-style, two shaped like the recovery→pulse scenarios in the briefs). Leave-one-run-out over the four non-initial runs, with *all* parameters refit inside each fold. Scores are reported separately for random-style and scenario-style folds.

**Recovering σ.** Local scores were far above the leaderboard's (0.84 vs 0.51 on one system). The Public score of each early submission was treated as a measurement, and I solved per system for a scalar `c` such that σ = c · std(data) makes the cross-validated errors reproduce it. `c` came out between 0.12 and 0.78 (much tighter than the data spread I had assumed), and the calibrated local scores tracked Public scores with r ≈ 0.75. This is a per-system point estimate, not the organizers' σ, and it is noisy.

**Models.**
- *Baselines.* The kit's one-step linear state-space model, and "direct" ridge regression from action-history features (EMAs at several half-lives, pairwise products, decayed initial state) to observables. The one-step model compounds error across 4,000 steps and was worse than persistence on a held-out run (0.435 vs 0.507). Direct ridge plateaued around 0.55–0.60.
- *GRU forecaster.* Inputs are min-max-scaled actions only, with no observation feedback after t = 0. The hidden state is initialized as `tanh(W·x0 + b)` from the normalized initial reading. Hidden size 32–64, linear read-out. Trained full-batch on entire rollouts (free-running by construction) with σ-weighted smooth-L1 (β = 0.05), Adam (lr 3e-3, weight decay 1e-4, cosine schedule, gradient clip 1.0), 1,500 epochs. Trained in PyTorch, exported to a ~40-line NumPy GRU (max |Δ| < 5e-4 vs torch). Ensembles average 2–10 nets.
- *Structural findings.* A per-observable cross-run check, correlating each output across independent runs at the same tick, showed reservoir inflow is action-independent (correlation 0.995). It fits a single sinusoid, period ≈ 67.8 ticks, fitted on ticks ≤ 256 and checked on ticks 256–450 (inflow score 0.93 out of sample). Replacing the network's inflow with the closed form took reservoir from 0.705 to 0.788 (+0.084).
- *Differentiable simulator (social contagion).* Per community: adopters, interest queue, cooldown pool and eligible population, with a shared onboarding workforce whose capacity falls with total adopters, plus credibility, incentive-expectation and cross-community memory states behind switchable gates. About 25 scalar parameters, fit end-to-end on free-running rollouts (Adam, lr 3e-2). The gate pair was chosen only by held-out score: credibility + expectations 0.572/0.586 (random/scenario folds), against 0.458/0.446 and 0.536/0.574 for the other pairs, and 0.496/0.575 for the best GRU. A 0.6 physics + 0.4 GRU blend (weight from a held-out grid) scored 0.655 on Public against 0.592 for the GRU.
- *Long-hold weighting.* The score receipts report two bands, sustained episodes ("id") and the other three categories ("extrapolation"). Sustained-band scores were the weakest almost everywhere: supply chain 0.575, market 0.621, traffic 0.626, power grid 0.681, against 0.73–0.87 on the other bands for those four systems. Upweighting training ticks at least 30 steps into a constant-action hold (weight 4×) improved held-out deep-hold scores on scenario folds and moved traffic 0.812 → 0.821.

## Negative results

| Attempt | Held-out signal | Public outcome |
|---|---|---|
| Per-observable blend of GRU with ridge (regression weights 0.6–0.7 on price/volume) | +0.004 / +0.033 | market 0.700 → **0.605**. Ridge drifts *inside* the data range over 4,000 steps, so the range-based drift check passed. |
| Physics + GRU for supply chain | +0.15 | 0.745 → **0.532** |
| Bigger GRU (H64, 3,000 epochs) and 10-net ensemble on social contagion | positive | 0.592 → 0.541 and 0.551 |
| Agent-built structured model + GRU correction (teacher-forced fit) on social contagion | 0.525 vs 0.532 for the GRU | 0.592 → 0.480 |
| Metric-aligned fine-tuning (L1, then the score itself) | wins on 3 systems | wildlife 0.799 → 0.794 |
| Long-hold weighting | mixed | up on traffic (0.812 → 0.821) and market (+0.001); down on epidemic, reservoir and supply chain (0.745 → 0.672) |
| Seasonal clock inputs (sin/cos) for epidemic | rhythm real but ≈0.3σ | not submitted; held-out 0.698 → 0.654 (random folds), 0.686 → 0.558 (scenario folds) |
| Physics models for traffic, market, power grid | not competitive | not submitted |

Two agent-produced results also failed audit: a reservoir "0.75" that came from refitting simulator parameters on the held-out run (0.63 after fixing the leak), and an epidemic "physics" candidate whose final output turned out to be a GRU with the physical branch unused.

## What I'd do differently

1. **Spend the experiment budget on the schedules that get scored.** I used it on broad random and structured coverage before knowing which episodes mattered. About 76 of 2,000 steps per system remained when the receipts showed sustained-hold episodes were the main loss, far too few to collect long holds.
2. **Validate at the target horizon.** Five runs of ≤ 510 steps make cross-validation noisy (roughly ±0.03 per system) and biased toward short-horizon behaviour. Its sign disagreed with Public on roughly half of my candidate uploads, so Public feedback was the only reliable judge and I had only three slots per system per day.
3. **Prefer structure over capacity.** The two largest single gains (inflow, +0.084; the social-contagion simulator, +0.06) came from identifying structure, not from bigger networks.

## Process

- All folds refit all parameters. Any gain claimed by an agent was independently re-run before use.
- Each candidate had a 4,000-step check (fraction of steps outside the research range ±5%, fraction flat) and a fresh-process 40 × 4,000 contract test on Python 3.12 / NumPy 2.3.5. One model was rewritten for a 17× speed-up to remove timeout risk.
- The Final submission contains only files byte-identical to uploads that had already scored their numbers on Public.

## Code

Research code written under time pressure and lightly cleaned, not a polished library.

```
gru/           GRU forecaster
  train_gru.py      4-fold held-out CV:  python gru/train_gru.py <system> <hidden> <seeds> <epochs>
  final_fit.py      train on all runs and export weights for the NumPy predictor
  predict.py        NumPy inference (what the grader runs)
  ensemble.py       mixed-size ensembles, CV and final fit
  metric_*.py       score-aligned fine-tuning (L1 warm-up, then the score itself)
  hold_*.py         long-hold weighting;  clock_cv.py: seasonal-input ablation
  train_gru2.py     EMA-input variant (with predict_ema.py)
diffsim/       differentiable simulators
  social.py         social-contagion simulator with switchable memory gates, plus CV
  social_*.py       restarts, physics/GRU blend weights, final fit
  hospital.py, hosp_pigru.py   hospital-queue simulator and a GRU fed with its forecasts (exploratory, not shipped)
direct/, direct2/   ridge action-history baselines
research/      common.py (loading, CV, sigma calibration helpers), fair_compare.py, blend_cv.py,
               time_diag.py (error by time and by observable), reservoir_inflow_fit.py
predictors/    NumPy inference as submitted: social_contagion_physics, social_contagion_blend,
               reservoir_inflow (wraps a GRU model and overwrites inflow with the fitted sinusoid)
collect_scenario.py, collect_structured.py   research-run collectors
```

The scripts assume the organizers' participant kit in the working directory (`client.py`, `fit.py`, `example_submission/predict.py`, `briefs.md`), five research runs per system in `research/`, and a calibrated `research/sigma_cal.json`. None of those are included. Trained weights are not included either. The hospital-queue model that shipped was built by an agent and isn't in this repository.

## Not included

Competition data, kit files, trained weights, the calibrated σ values, credentials and hidden scoring details.
