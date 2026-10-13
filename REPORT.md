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

*(to complete)*

## Modeling: three implementations, one model

*(to complete)*

## Limitations & next steps

*(to complete)*

## Generative AI use disclosure

*(to complete, honestly: what you used and how)*
