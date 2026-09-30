<p align="center">
  <img src="https://datasciencefestival.com/wp-content/uploads/2023/09/google-deepmind-logo.webp" alt="Google DeepMind" width="320">
</p>

# 10 Black Boxes. 40 Hours. 0.04 → 0.76.

> *Ten hidden simulators. Equations sealed. A budget of 2,000 experiments each, and one shot to predict 4,000 steps into the future with nobody telling you how you're doing.*

My entry to the **InstaLILY × Google DeepMind GroundTruth challenge**, Toronto, September 2026.

---

## The setup

Ten simulated worlds: **epidemic, market, traffic, power grid, supply chain, wildlife, reservoir, ad auction, social contagion, hospital queue.** You can poke each one through an API and watch a few noisy readings come back. That's it. The equations are hidden, and in every world *exactly two of three secret mechanisms are secretly switched on*. Nobody tells you which.

Then comes the exam: predict **4,000 steps** of a system you've barely touched, under action schedules you've never seen, with **zero feedback while forecasting**. Every step is scored as `1 / (1 + error/σ)`, and σ, the yardstick, is a secret too.

The clock on my side: **about 40 hours left** when I started.

## The climb

| Moment | Score |
|---|---|
| First upload: the organizers' starter model, one system | **0.039** overall |
| Every system covered for the first time | 0.496 |
| First neural state-space models | 0.686 |
| Reservoir discovery + physics for the hardest system | 0.73 |
| **Final submission** | **≈ 0.757** |

Final Public scores: ad auction **0.867** · traffic **0.821** · wildlife **0.799** · reservoir **0.788** · power grid 0.746 · supply chain 0.745 · epidemic 0.743 · hospital queue 0.711 · market 0.701 · social contagion 0.655.

From 45th place to the top tier of a 125-person field in a day and a half. Not the win. But here's how it went.

---

## Act I: My own scoreboard was lying to me

My models "scored" 0.84 in my tests. The leaderboard said **0.51**.

Something was badly wrong. So I turned the leaderboard into an instrument: I solved for the hidden σ, finding the yardstick at which my cross-validated errors reproduced each real score. It was **3-8× stricter** than I'd assumed. Every decision after that used the calibrated metric.

## Act II: The model that was worse than doing nothing

My first submission scored 0.387 on power grid. I checked it against a run it had never seen: worse than just repeating the initial reading. The starter model steps forward from its own last prediction, so tiny errors compound across 4,000 steps like a rounding error in a rocket trajectory.

The fix was a different *kind* of model: small **GRUs** that read the action schedule and carry their own hidden state, trained on whole free-running rollouts so they can't feed on their own mistakes. Trained in PyTorch, then rewritten in ~40 lines of pure NumPy, because the grader has no deep-learning libraries (difference under 1e-4).

**One upload. 0.525 → 0.686.** Epidemic went from 0.39 to 0.74.

## Act III: The number hiding in plain sight

Per-output breakdowns showed one weak spot: reservoir **inflow**. Across five independent runs, inflow was **identical at the same step** (correlation 0.995) no matter what I did to the controls. It wasn't a system response. It was a clock.

A pure sinusoid, period ≈ 67.8 steps. I fitted it on the first 256 steps, confirmed it on steps 256-450 it had never seen, and replaced the network's inflow with the formula.

**Reservoir: 0.705 → 0.788.** One insight, +0.084.

## Act IV: Physics beats the neural net

**Social contagion**: boom-and-bust adoption, onboarding queues, cooldown pools, shared staff, memory. Every neural variant got stuck near 0.59, and the score jumped around by ±0.05 depending only on the random seed.

So I wrote the brief's physics as a **differentiable simulator** in PyTorch: adopters, interest queue, cooldown pool, shared onboarding workforce, and three switchable memory mechanisms. It has **22 parameters**, and it can't memorize noise.

- It **beat the GRU on held-out runs.**
- Held-out selection pinned the hidden mechanisms: **credibility + incentive expectations.**
- Blended 60/40 with a GRU: **0.592 → 0.655** on the real leaderboard.

## Act V: The receipts told me where the bodies were buried

The score receipts split results into *sustained* vs. *other* episodes, and the truth came out: six systems were quietly bleeding points on **long constant holds**: supply chain **0.58**, market 0.62, traffic 0.63. My research runs never held a setting longer than ~180 steps, so the models had never seen how these systems settle.

Re-weighting training toward steps deep inside holds lifted **traffic 0.812 → 0.821** and market slightly.

---

## The rules I lived by (after breaking them)

- **Validation without leakage.** Every fold refits every parameter. One AI-generated "0.75" evaporated to **0.63** once the leak was removed.
- **4,000-step drift checks.** Caught two models that looked perfect on short runs and would have gone off the rails at the real horizon.
- **A grader harness:** fresh processes, 40 × 4,000 steps, Python 3.12 / NumPy 2.3.5, deterministic. One model was sped up **17×** to remove any timeout risk.
- **Only promote what has already scored.** The Final submission is byte-identical to files that had proven their score on the Public leaderboard.
- **Distrust every claim,** including my own and my AI agents'. I ran several AI coding agents in parallel and re-verified every "win" before shipping it.

## What blew up in my face

- A **regression blend** that looked +0.03 better in testing dropped market from **0.700 to 0.605** on the real board. Regression drifts over long horizons. I learned it publicly.
- **10-network ensembles** helped wildlife and hurt other systems. Short-run cross-validation often predicted the wrong sign.
- **Seasonal "clock" inputs** for epidemic: the rhythm was real but tiny, and it made the network overfit.
- A physics model for supply chain that scored **0.53 against a 0.74 baseline**, and an undo upload that saved the day.

## What I'd do differently

1. **Spend the experiment budget like it's gold.** It was the only source of truth, and I burned it on broad coverage before knowing what mattered. Long holds would have been worth more than any modeling trick.
2. **Validate at the target horizon.** Models that win at 460 steps can lose at 4,000.
3. **Structure beats capacity when data is scarce.** The biggest jumps all came from *understanding* a system (the inflow clock, the queue physics), not from making a network bigger.

---

**Stack:** Python · PyTorch (training) · NumPy/SciPy (inference) · differentiable simulation · GRU state-space models · AI coding agents

*Competition data, credentials and hidden-scoring details are not included, out of respect for the organizers' rules.*
