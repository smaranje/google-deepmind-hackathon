<p align="center">
  <img src="https://datasciencefestival.com/wp-content/uploads/2023/09/google-deepmind-logo.webp" alt="Google DeepMind" width="320">
</p>

# InstaLILY × Google DeepMind GroundTruth challenge: my approach

Notes on my entry to the GroundTruth research and forecasting challenge (Toronto, September 2026). I joined with about 40 hours left, worked with AI coding agents that wrote and ran most of the code, and drove the process myself: choosing what to try, reading the leaderboard feedback, and deciding what to upload.

## The challenge

There are ten hidden simulators: epidemic, market, traffic, power grid, supply chain, wildlife, reservoir, ad auction, social contagion and hospital queue. For each one you get 2,000 experiment steps through an API and see a few noisy readings. You then submit a `predict()` function that forecasts 4,000 steps ahead from a starting reading and a full action schedule, with no feedback while forecasting. It runs offline with NumPy/SciPy only.

Each step is scored as `1 / (1 + |error| / σ)`, where σ is a hidden scale set by the organizers. The overall score is the mean over the ten systems. In every system, two of three described "memory" mechanisms are active, and you aren't told which.

## Results

| Point in time | Public score (mean over 10 systems) |
|---|---|
| First upload (organizers' starter model, one system only) | 0.039 |
| First model for all ten systems | 0.496 |
| After switching to GRU models | 0.686 |
| End of the second day | about 0.73 (rank 22 of 125 on the published board) |
| Final submission (best model per system) | about 0.757 expected, based on Public scores |

The Final results weren't published when I wrote this, and Final uses different hidden episodes from Public, so the real number will differ a little. I'll update this page when it's out.

Per-system Public scores of the final submission: ad auction 0.867, traffic 0.821, wildlife 0.799, reservoir 0.788, power grid 0.746, supply chain 0.745, epidemic 0.743, hospital queue 0.711, market 0.701, social contagion 0.655.

## What worked

- **Working out the hidden scoring scale.** My own test scores were far higher than the leaderboard's (0.84 locally against 0.51 on the board). I solved for the value of σ that made my cross-validated errors match each real score. It was about 0.12–0.78 times the spread of the data, much stricter than I had assumed. After that, model comparisons were much more reliable.
- **GRU models.** Linear and ridge models stopped improving at about 0.55–0.60. Small GRUs (hidden size 32–64) that read the action schedule and carry a hidden state, trained on whole free-running rollouts, lifted the score a lot (epidemic went from 0.39 to 0.74). They're trained in PyTorch and re-implemented in plain NumPy for the grader.
- **A pattern in reservoir inflow.** In five separate runs, inflow was identical at the same step no matter what the controls did (correlation 0.995). It fits a single sine wave with a period of about 68 steps. Replacing the model's inflow with that formula raised reservoir from 0.705 to 0.788.
- **A physics model for social contagion.** Neural models stalled around 0.59. A small differentiable simulator with queues, a cooldown pool and memory mechanisms (22 parameters), blended 60/40 with a GRU, scored 0.655. Held-out tests pointed to credibility plus incentive expectations as the active mechanisms.
- **Reading the score receipts.** They split each score into sustained (long constant hold) episodes and the rest. Most systems lost points on long holds, because my research runs never held a setting for longer than about 180 steps. Training that puts more weight on long holds raised traffic from 0.812 to 0.821.

## What didn't work

- A blend that added ridge regression to the GRU looked better in testing but dropped market from 0.700 to 0.605 on the real board. Regression models drift over 4,000 steps.
- Larger networks and 10-network ensembles helped some systems and hurt others. Short-run cross-validation often predicted the wrong direction.
- Seasonal clock inputs for epidemic overfit.
- A physics model for supply chain scored 0.532, against 0.745 for the model it was meant to replace, and I had to revert it.
- Physics models for traffic, market and power grid didn't beat the GRUs in the time I had.

## Process

- Every cross-validation fold refits all parameters. One AI-generated "0.75" for reservoir dropped to 0.63 once a data leak was removed.
- Each model got a 4,000-step check for drift outside the data range and for flat-lining, and a runtime test at the full 40 × 4,000 steps under the time limit.
- The Final submission contains only files that had already scored their numbers on the Public leaderboard.

## Lessons

1. Deciding what data to collect is the most important step. Spending part of the budget on very long holds would have been worth more than any modeling trick.
2. Test at the length you'll be scored on. Models that win on 460-step runs can lose on 4,000-step episodes.
3. With little data, structure beats size. The biggest gains came from understanding a system, not from bigger networks.

## How this was built

I worked with AI coding agents throughout. They wrote and ran most of the code and analysis, and I directed the work, judged the results and did the uploads. I re-checked each claimed improvement against the real leaderboard before relying on it, since the agents sometimes overstated results.

Stack: Python, PyTorch (training), NumPy/SciPy (inference).

Competition data, credentials and hidden scoring details aren't included.
