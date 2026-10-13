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

*(to complete)*

## Limitations & next steps

*(to complete)*

## Generative AI use disclosure

*(to complete, honestly: what you used and how)*
