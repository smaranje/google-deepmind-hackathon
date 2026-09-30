<p align="center">
  <img src="https://datasciencefestival.com/wp-content/uploads/2023/09/google-deepmind-logo.webp" alt="Google DeepMind" width="320">
</p>

# Predicting what happens when you change the controls

**Ten hidden systems. A small experiment budget. Forecast 4,000 steps ahead with no feedback.** My entry to the Google DeepMind GroundTruth challenge (Toronto, September 2026).

**In short**
- The task: learn how a system responds to decisions (staffing, tolls, vaccination rates, water release) from under 2,000 experimental steps, then forecast a long sequence of decisions nobody has tried yet.
- What I built: a neural network with memory (a GRU) per system, plus physics-style models where the data was too thin for a network to work out the structure.
- Result: mean Public score **0.04 → 0.735** in roughly 40 hours (rank 21 of 125 on the last published board I saw). My final submission is expected to score about **0.757** on Public. Final results aren't out yet.

## Why this kind of forecasting matters

Real decisions are rarely "what will happen if nothing changes?" They are "what happens if we do X, then Y, then Z?" A hospital deciding when to add overtime before a surge, a grid operator choosing when to dispatch reserves, or a water utility balancing supply against water quality all need a model that answers that *before* they try it on real people.

Building such a model from limited experiments is a standard engineering problem (system identification). It is hard because these systems have **hidden memory**: queues, stock levels, fatigue, credibility. A decision made today changes what happens hundreds of steps later, and you can't see the hidden state directly.

One honest caveat: the organizers built these ten systems as **invented mathematical simulators**, and they say they are not suitable for real decisions. What carries over to real work is the method, not the models.

## The ten systems

Each system has named controls and a few noisy readings. Equations and parameters are hidden. In each one, two of three described "memory" mechanisms are switched on, and you aren't told which.

| System | The real-world question it resembles | Controls → what gets forecast | What worked best | Public score |
|---|---|---|---|---|
| **Ad auction** | How much to bid and spend to win impressions and conversions | bid, budget cap, targeting breadth → win rate, spend, conversions | 10-network GRU ensemble | **0.867** |
| **Traffic** | How tolls, lane closures and signal timing change congestion | 6 controls (tolls, ramp metering, closures…) → flow and speed on two routes | GRU trained with extra weight on long steady periods | **0.821** |
| **Wildlife** | How hunting quotas and habitat protection change predator and prey numbers | quota, habitat protection, corridor access → prey and predators in two regions | 10-network GRU ensemble | **0.799** |
| **Reservoir** | How to release and withdraw water while keeping quality up | release rate, irrigation, withdrawal depth, aeration → level, inflow, outflow, quality | GRU ensemble + a fitted seasonal inflow curve | **0.788** |
| **Power grid** | How to keep frequency stable while shifting demand and dispatching reserves | price signal, reserve dispatch, charging, interconnector → load, frequency, renewable share | GRU blended with a simple regression | 0.746 |
| **Supply chain** | How ordering and staffing move inventory and shipments | order size, lead time, product mix, effort, maintenance → shipments, supplier and retail stock | action-history regression with physical limits (no negative stock, supplier cap) | 0.745 |
| **Epidemic** | Which mix of school closure, masks and vaccination limits cases and hospital strain | 3 policies → daily cases, hospital load | GRU | 0.743 |
| **Hospital queue** | When staffing, overtime and scheduling keep waits down without burning out staff | 6 controls → wait time, queue length, discharges | physics-style model + GRU | 0.711 |
| **Market** | How interest rates and a transaction tax move price, volume and liquidity | interest rate, tax → price, volume, order-book depth | GRU trained with extra weight on long steady periods | 0.701 |
| **Social contagion** | How seeding, incentives and outreach spread adoption across two communities | seeding, incentive, bridge outreach → adopters in each community | **physics simulator + GRU blend** | 0.655 |

How to read the scores: 1.0 would be a perfect forecast at every step. Each step scores `1 / (1 + error / σ)` with a hidden tolerance σ, so **0.75 means a typical error of about a third of that tolerance**. The final column is the Public leaderboard score of the model in my Final submission.

## How it works

```
Research runs        Learn the dynamics                    Forecast
(holds, pulses,  →   Controls ─► GRU (hidden memory) ─►    4,000 steps of readings
 recoveries)         first reading sets the start state    from the first reading
                     └─ + physics/structure where data is thin   and the control schedule only
```

**1. Collect the right experiments.** The scoring uses four kinds of episodes: long steady operation, reordered actions, repeated stress with rest, and controls changed alone or together. I collected runs shaped like those, including the "recovery" and "pulse" settings each brief describes.

**2. Work out the scoring tolerance.** My local scores were far above the leaderboard's (0.84 vs 0.51 on one system). Treating each leaderboard score as a measurement, I solved for the hidden tolerance σ that made my cross-validated errors reproduce it. It was about 0.12–0.78 times the spread of the data, much stricter than I had assumed. Every model comparison after that was more trustworthy.

**3. Learn the dynamics with a GRU.** A GRU is a small neural network that carries a hidden state, which is exactly what these systems need to represent their queues, stocks and fatigue. It takes the schedule of controls as input and predicts every reading for all 4,000 steps. Its starting state comes from the first reading, and it never sees a real reading again. That matters: a model that feeds on its own predictions piles up small errors over thousands of steps. The organizers' starter model did this, and it scored 0.387 on power grid, worse than repeating the first reading.

**4. Add structure where the network can't find it alone.** Two examples:
- *Reservoir inflow* is a fixed seasonal cycle that ignores the controls (the same at each step across independent runs, correlation 0.995). Replacing the network's inflow with a fitted sine wave (period about 68 steps) lifted reservoir from 0.705 to 0.788.
- *Social contagion* has boom-and-bust adoption, an onboarding queue, a cooldown period for people who lose interest, and shared staff. I wrote these as a differentiable simulator (about 25 parameters) trained end to end on whole runs. Blended 60/40 with a GRU, it scored 0.655, against 0.592 for the best network. Held-out tests also pointed to which two hidden mechanisms are active: credibility and incentive expectations.

**5. Ship only what has been proven.** Every candidate got a 4,000-step stability check and a full-scale timing test. The Final submission contains only files that had already scored their numbers on the Public leaderboard.

## What moved the score

| Change | Effect |
|---|---|
| Switching from one-step models to GRUs trained on whole runs | mean 0.496 → 0.686 (epidemic 0.39 → 0.74) |
| Seasonal inflow curve for reservoir | 0.705 → 0.788 |
| Physics simulator blended with a GRU for social contagion | 0.592 → 0.655 |
| Extra training weight on long steady periods (traffic) | 0.812 → 0.821 |

## What didn't work

| Attempt | Local signal | Real outcome |
|---|---|---|
| Blending a regression model into the GRU for market | looked better | market 0.700 → **0.605**. Regression drifts over 4,000 steps, and my drift check missed it because it stayed inside the data's range. |
| Physics + GRU for supply chain | looked much better | 0.745 → **0.532** |
| Bigger networks and 10-network ensembles for social contagion | looked better | 0.592 → 0.541 and 0.551 |
| Seasonal "clock" inputs for epidemic | rhythm was real but tiny | not submitted; the network overfit it in testing |

Two AI-generated results also failed my audit. One "0.75" for reservoir came from a data leak (0.63 once removed). One "physics" model for epidemic turned out to output a plain GRU, with the physics branch unused.

## Limits and lessons

- **Little data.** Five runs of at most about 510 steps per system, against 4,000-step scoring. Local cross-validation was noisy (roughly ±0.03 per system) and its sign disagreed with the real score on about half my candidate uploads, so the leaderboard was the only reliable judge.
- **Experiment budget.** I spent it on broad coverage before I knew which episodes mattered. The receipts later showed most systems lost points on **long steady periods**, and only 76 of the 2,000 steps per system were left to collect them.
- **The biggest gains came from structure, not size:** the inflow curve (+0.084) and the social simulator (+0.06), not larger networks.
- **Final results** use different hidden episodes, so the real number will differ from 0.757 by up to about 0.01.

## How this was built

I used AI coding assistants for implementation and analysis, working with them like a pair. I set the strategy and priorities, decided which ideas to pursue (for example, going back to the rules and switching to recurrent networks when linear models plateaued), chose what to upload, and used the leaderboard results to decide what to keep. Any gain an assistant reported was re-run independently before I relied on it.

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

The scripts assume the organizers' participant kit in the working directory (`client.py`, `fit.py`, `example_submission/predict.py`, `briefs.md`), five research runs per system in `research/`, and a calibrated `research/sigma_cal.json`. None of those are included. Trained weights are not included either. The hospital-queue model that shipped isn't in this repository.

## Not included

Competition data, kit files, trained weights, the calibrated σ values, credentials and hidden scoring details.
