# Assignment 1 Report

- **Name**: Jaime Fernandez
- **Student ID**: 18632
- **Email**: jfernandezr.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

NFL game results from the `nflverse/nfldata` project (`games.csv`,
https://github.com/nflverse/nfldata). I froze a copy in `data/nfl_games.csv`
(seasons 1999–2025, only games already played) so the pipeline is reproducible:
the original file is updated every week.

One row = one NFL game. The frozen file has 7,276 rows and 39 columns (teams,
season and week, days of rest for each team, roof and surface, temperature and
wind, starting quarterbacks, betting lines, final score...). I kept only
regular-season games and dropped ties (15), leaving **6,952 games**. I picked it
because the outcome is a clear yes/no, the file has a natural time order, and
most of the useful signal (recent form, rest, strength) has to be built by me
from past games.

## Business / real-life framing

**Question:** will the home team win this regular-season game?

**Decision it supports:** a coach uses the prediction to decide whether to rest
starters. If the model is confident the home team wins, rest them; otherwise,
play them.

**Two errors, two different costs** (the numbers are invented for the exercise,
not measured):

| Error | What happens | Cost |
|---|---|---|
| False positive | Model says "win", I rest starters, and the team **loses** | 3 |
| False negative | Model says "no win", I play starters, and the team would have won anyway | 1 |

**Decision threshold.** Saying "win" costs `(1 - p) * 3` in expectation and
saying "no win" costs `p * 1`. Saying "win" is better when `(1 - p) * 3 < p`,
i.e. when `p > 3 / (3 + 1) = 0.75`. So the decision threshold is **0.75, not
0.5**. The dashboard slider moves this threshold and shows how the confusion
matrix and the total cost change.

**What this implies for the pipeline:**

- **Target:** `home_win = 1` if the home team's score is higher. Ties are dropped
  so the target is binary.
- **Time-based split, not random.** The model must predict games it has not seen,
  in the future. A random split would train on later seasons to "predict"
  earlier ones. Train: 1999–2021 (5,868 games), validation: 2022 (269 games,
  used only to tune regularization and learning rate), test: 2023–2025
  (815 games, used only for the final comparison).
- **Leakage removed:** final scores, `result`, `total` and `overtime` (they
  describe the outcome), and all betting lines and odds (the market's opinion,
  not sports data; using them would mostly copy the market's forecast).
- **Baseline:** always predict "home team wins". Home win rate is 56.4% in train
  and 54.2% in test, so the baseline's test accuracy is 54.2%.
- **Metric:** accuracy alone is misleading here, because the costs are
  asymmetric. The model is judged by total business cost at the 0.75 threshold,
  with accuracy and log loss as secondary checks.

## Data preparation & feature engineering

**Rule that drives everything: a feature for game *i* may only use games played
before *i*.** The raw file has no "team strength" column, so I built it from past
results. For every team and every game I computed the team's average point margin
(points scored minus points allowed) over its previous games, using
`shift(1)` so the current game is excluded, and then took **home minus away**.

| Feature | What it measures | Why |
|---|---|---|
| `pd5_diff` | Home minus away average point margin, last 5 games | Recent form |
| `pd16_diff` | Same, last 16 games (about one season) | Underlying strength, less noisy than 5 games |
| `rest_diff` | Home rest days minus away rest days | A team with more rest should be fresher |
| `div_game` | 1 if both teams share a division | Rivals know each other, so games may be closer |

Correlation with winning in the training set: `pd5_diff` 0.28, `pd16_diff` 0.30,
`rest_diff` 0.02, `div_game` -0.03. The two margin features carry the signal;
rest and division are kept because they have a clear football logic, but I expect
their weights to be small and I report that rather than hide it.

**Proof of no leakage.** `check_no_future_leakage` in `train.py` scrambles the
scores of every game from the first test day onward and checks that the features
of games on that day do not change, while the features of later games do. It
runs every time `main.py train` runs.

**Data-quality decisions**

- Both margin features are undefined for a team's very first game in the data
  (1999 week 1). I fill them with 0 (an average team). This affects only the
  first game of each team.
- `temp` and `wind` are missing in 1,982 of the 6,952 games, mostly because
  roofed stadiums do not record them. `surface` has nine spellings, including
  `grass` and `grass ` (with a trailing space). I tried adding indoor/outdoor,
  grass/turf, temperature and wind (imputing outdoor medians from the training
  set). On an earlier four-feature version, validation accuracy went from 0.613
  to 0.602 and log loss did not improve, so I **left the weather and surface
  features out**.
- **Scaling:** features are standardized with a `StandardScaler` fitted on the
  training set only, then applied to validation and test. Scaling matters here
  because the manually written gradient descent converges poorly on features
  with different scales.

**What I tried and did not keep.** A longer or shorter window on its own, a
season-to-date margin, and an Elo-style rating all gave similar results. I chose
the 5-plus-16 game combination because it did better on the 2022 validation
season (accuracy 0.639 against 0.613 for a 5-game win rate plus margin) and is
easy to explain. Elo performed comparably but is harder to explain and defend.
In the interest of honesty: my exploration scripts also printed test-set
numbers, so the test set is not perfectly untouched. The final choice was made
on validation results and simplicity.

## Modeling: three implementations, one model

**Model: logistic regression**, because the target is binary (home win yes/no)
and the output is a probability I can compare to the 0.75 cost-based threshold.
`probability = sigmoid(X @ w + b)`, trained by minimizing mean binary
cross-entropy plus an L2 penalty `(lambda / 2) * sum(w^2)` on the weights (not the
bias).

**Three implementations on the same split and features:**

1. **scikit-learn** `LogisticRegression` (lbfgs solver).
2. **Manual PyTorch loop** (Session 5): raw tensors, a hand-written BCE loss,
   `loss.backward()`, an update `w -= lr * w.grad` inside `torch.no_grad()`, and
   `grad.zero_()` after every step.
3. **Standard PyTorch**: `nn.Sequential(nn.Linear, nn.Sigmoid)`, `nn.BCELoss`,
   `torch.optim.SGD`.

**What I set equal so the three could be compared fairly:**

- **Same objective.** scikit-learn's `C` is the inverse of the penalty. With
  `n` training rows, the equivalent for a mean loss is `lambda = 1 / (C * n)`.
  In method 2 I add the penalty to the loss explicitly; in method 3 I use
  `weight_decay=lambda` on the weights only (the bias sits in a separate
  parameter group with no decay). `weight_decay` adds `lambda * w` to the
  gradient, which is the gradient of the penalty.
- **Same optimizer and starting point.** Both PyTorch versions use plain
  full-batch gradient descent (no momentum, no Adam), 3,000 epochs, and weights
  and bias initialized at zero (`nn.Linear` would otherwise start random).
- **Same inputs.** Standardized features, same split.

**Hyperparameters (the only two I tuned, on the 2022 validation season).**
`C` from {0.001 ... 100}: validation log loss is flat from `C = 1` upwards
(0.64484 vs 0.64483), so I chose `C = 1.0`, the strongest regularization within
noise (differences under 5e-5 are treated as ties). Learning rate from
{0.1, 0.5, 1, 2}: all four reach the same validation loss, so training converges
for each, and I chose the most cautious one, `0.1`. In short, with 4 features and
5,868 training games, regularization and learning rate barely matter.

**Results on the test set (2023-2025, 815 games):**

| | Accuracy @0.5 | Log loss | AUC | Cost @0.75 | False pos. | False neg. | Games rested |
|---|---|---|---|---|---|---|---|
| scikit-learn | 0.6319 | 0.6386 | 0.6896 | 418 | 19 | 361 | 100 |
| Manual PyTorch | 0.6319 | 0.6386 | 0.6896 | 418 | 19 | 361 | 100 |
| nn.Module + optim | 0.6319 | 0.6386 | 0.6896 | 418 | 19 | 361 | 100 |
| Baseline: always predict home win | 0.5423 | 0.6905 (train win rate) | - | 1,119 (always rest) | - | - | 815 |
| Baseline: never rest starters | - | - | - | 442 | 0 | 442 | 0 |

**Do the three agree? Yes.** All three learn the same coefficients to four or
five decimals (standardized features; order is `pd5_diff`, `pd16_diff`,
`rest_diff`, `div_game`, bias):
`0.2719, 0.4715, 0.0446, -0.0614, 0.2857`. The largest difference in predicted
probability between any two methods on the test set is 1.5e-6, which comes from
`float32` (PyTorch) against `float64` (scikit-learn) and from different stopping
rules (lbfgs against 3,000 gradient steps). I expected agreement: the logistic
objective with an L2 penalty is convex, so there is one optimum, and I forced
the three methods to minimize the same function from the same point. The three
could have disagreed if I had left the penalty out of one of them, used Adam in
one and plain gradient descent in the other, forgotten `grad.zero_()`, or skipped
the scaling. I did not need to debug a disagreement, but note that these three
methods are the same model with different bookkeeping, so agreement shows I
implemented the maths correctly, not that the model is good.

**Reading the numbers against the business framing.**

- The model beats the naive baseline on every statistic: accuracy 63.2% against
  54.2%, and log loss 0.639 against 0.691.
- The coefficients make football sense: the longer-run margin (`pd16_diff`,
  0.47) matters most, then recent form (`pd5_diff`, 0.27). Rest and division
  barely matter, as the correlations suggested.
- **The business gain is real but small.** At the 0.75 threshold the model rests
  starters in only 100 of 815 games, and 19 of those are losses. Total cost is
  418, against 442 for the best naive policy (never rest starters), a reduction
  of about 5%. "Always rest" costs 1,119, much worse. Because the home team wins
  only 54% of the time and the threshold is 0.75, the sensible naive policy is
  to never rest, and that is the bar the model has to clear.

## Limitations & next steps

1. **The business gain is small and not statistically solid.** At the 0.75
   threshold the model costs 418 against 442 for "never rest", a saving of 24.
   A bootstrap over the 815 test games gives a 95% interval of about -9 to +54
   for that saving, so I cannot claim the model beats the simple policy with
   confidence (it won in about 92% of resamples). With 815 games, accuracy also
   carries a standard error of about 1.7 points. *Next step:* evaluate on more
   seasons as they are played, and report intervals rather than single numbers.
2. **The costs (3 and 1) are invented**, and the 0.75 threshold follows directly
   from them. If the real cost of a lost game were, say, 5 times a fatigue risk,
   the threshold would be 0.83. The dashboard lets you change the costs, but the
   true values would have to come from the club (standings impact, injury
   data). *Next step:* ask a team analyst for realistic cost estimates.
3. **The model overestimates home advantage in recent seasons.** It predicts a
   56.9% home win rate on average in the test set, against 54.2% observed. The
   intercept was learned mostly from older seasons (home win rate 56.4%).
   Among the 100 games where it recommends resting, the home team did win 81%,
   so the decision area is reasonably calibrated, but the overall bias is
   visible. *Next step:* weight recent seasons more, or recalibrate the
   intercept on the most recent seasons.
4. **The features ignore who the opponent was and who is playing.** A margin of
   +10 against weak teams counts the same as +10 against strong ones, and the
   data has the starting quarterback of each game (`home_qb_id`) but I did not
   use it, though a quarterback change is one of the biggest swings in a team's
   strength. *Next step:* opponent-adjusted margins (or an Elo-style rating,
   which I tested and which performed comparably) and a "starting quarterback
   changed" flag.
5. **One validation season is a noisy basis for tuning.** 2022 has 269 games,
   and the validation loss was flat for every regularization strength above
   `C = 1` and for every learning rate I tried, so tuning decided almost nothing.
   *Next step:* rolling-origin cross-validation across several seasons.
6. **A linear model cannot capture interactions** (for example, rest mattering
   more when travelling). Rest and division had near-zero weights, and I kept
   them because of their football logic rather than their evidence. *Next step:*
   drop them or test interactions, and compare with a tree-based model, which
   is outside this assignment's scope.
7. **The decision scenario is stylized.** Real rest decisions depend on
   standings, injuries and playoff seeding, and are made player by player, not
   game by game. The model estimates a win probability; turning that into a
   rotation policy needs football input I do not have.
8. **The test set is not perfectly clean.** While exploring features I printed
   test-set numbers as well as validation ones (see the feature engineering
   section). I chose features using validation results and simplicity, but a
   fully strict workflow would have hidden the test set until the end.

## Generative AI use disclosure

*(to complete, honestly: what you used and how)*
