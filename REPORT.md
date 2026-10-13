# Assignment 1 Report  
**Name:** Jaime Fernández  
**Student ID:** 18632  
**Email:** jfernandezr.ieu2022@student.ie.edu  
**Group:** BBADBA 5A  

---

## Dataset

For this assignment, I used NFL game results from the *nflverse/nfldata* project. I saved a frozen copy of the dataset (seasons 1999–2025, only games already played) so the pipeline is reproducible. This frozen copy has **7,276 games** and 39 columns.

Each row represents one NFL game. The file includes information such as teams, week, rest days, roof/surface, temperature, wind, starting quarterbacks, betting lines, and final scores. After keeping only regular‑season games and removing ties, the final dataset contains **6,952 games**.

I chose this dataset because the outcome is binary (home win or not), the data has a natural time order, and most useful signals (recent form, strength, rest) must be engineered from past games.

## Business Framing

**Question:** Will the home team win this regular‑season game?

**Decision:** A coach uses the prediction to decide whether to rest starters.  
- If the model is confident the home team will win, starters rest.  
- Otherwise, starters play.

There are two types of errors with different costs:

| Error | What happens | Cost |
|------|--------------|------|
| **False positive** | Model says “win”, starters rest, team loses | **3** |
| **False negative** | Model says “no win”, starters play, team would have won anyway | **1** |

If the model’s predicted win probability is \(p\), saying “win” has expected cost \((1 - p) \cdot 3\) and saying “no win” has expected cost \(p \cdot 1\). Saying “win” is better when:
\[
(1 - p) \cdot 3 < p \quad \Rightarrow \quad 3 - 3p < p \quad \Rightarrow \quad 3 < 4p \quad \Rightarrow \quad p > \frac{3}{3 + 1} = 0.75.
\]
So the decision threshold is **0.75**, not 0.5.

### Pipeline implications

- **Target:** `home_win = 1` if the home team scores more points.  
- **Time‑based split:**  
  - Train: 1999–2021 (5,868 games)  
  - Validation: 2022 (269 games)  
  - Test: 2023–2025 (815 games)  
- **Leakage removed:** final scores, results, totals, overtime, and all betting lines.  
- **Baseline:** always predict “home win” → test accuracy **54.2%**.  
- **Main metric:** total business cost at threshold 0.75, with accuracy and log loss as secondary checks.

## Feature Engineering

Key rule: **a game’s features can only use information from earlier games**.

Since the dataset doesn’t include team strength, I built it using point margin (points scored minus points allowed) from previous games. For each team and each game, I computed the average point margin over its past games, using a `shift(1)` so the current game is excluded, and then took home minus away. This guarantees that no future information leaks into the features.

To further check for leakage, `check_no_future_leakage` in `train.py` scrambles the scores of every game from the first test day onward and verifies that features for games on that day do not change, while features for later games do. This check runs every time the training script is executed.

Final features:

| Feature | What it measures | Why |
|---------|------------------|-----|
| **pd5_diff** | Home minus away average margin over last 5 games | Recent form |
| **pd16_diff** | Same over last 16 games | More stable strength |
| **rest_diff** | Home rest days minus away rest days | More rest → fresher |
| **div_game** | 1 if both teams are in the same division | Rivalry games often closer |

Correlations with winning (train):  
- pd5_diff: **0.28**  
- pd16_diff: **0.30**  
- rest_diff: **0.02**  
- div_game: **−0.03**

The margin features carry most of the signal; rest and division are kept for football logic, but their weights are expected to be small.

### Data quality decisions

- First games of each team have no past margin → filled with 0 (treated as an average team).  
- Temperature and wind are missing in many games, mostly in roofed stadiums, and **surface** has nine inconsistent spellings (e.g. `"grass"` and `"grass "`). I tried adding indoor/outdoor, grass/turf, temperature and wind (with imputation), but validation accuracy and log loss did not improve, so I left weather and surface features out.  
- Features were standardized using a scaler fitted only on the training set, then applied to validation and test. Scaling matters because manual gradient descent converges poorly when features have very different scales.

I tested other window lengths, a season‑to‑date margin, and an Elo‑style rating. They gave similar results. I chose the 5+16 game combination because it performed slightly better on the 2022 validation season and is easy to explain. Elo performed comparably but is harder to defend.

## Modeling

I used **logistic regression**, since the target is binary and the output is a probability that can be compared to the 0.75 threshold. The model is:
\[
\text{probability} = \sigma(Xw + b),
\]
trained by minimizing mean binary cross‑entropy plus an L2 penalty \(\frac{\lambda}{2} \sum w^2\) on the weights (not the bias).

I implemented the same model in three ways:

1. **scikit‑learn LogisticRegression (lbfgs solver)**  
2. **Manual PyTorch loop** (raw tensors, hand‑written BCE loss, `loss.backward()`, manual weight update inside `torch.no_grad()`, and `grad.zero_()` after each step)  
3. **PyTorch nn.Sequential + optim.SGD** (a standard `nn.Linear` + `nn.Sigmoid` with `nn.BCELoss` and `torch.optim.SGD`)

### Making the three implementations comparable

To compare them fairly, I matched:

- **Same objective.** scikit‑learn’s `C` is the inverse of the penalty. With \(n\) training rows, the equivalent mean‑loss penalty is:
  \[
  \lambda = \frac{1}{C \cdot n}.
  \]
  In the manual PyTorch version I add the L2 term explicitly to the loss; in the `nn.Module` version I use `weight_decay = lambda` on the weights only (the bias is in a separate parameter group with no decay). `weight_decay` adds \(\lambda w\) to the gradient, which is the gradient of the penalty.  
- **Same optimizer and starting point for the PyTorch models.** Both PyTorch versions use full‑batch gradient descent (no momentum, no Adam), **3,000 epochs**, and weights and bias initialized at zero. scikit‑learn uses `lbfgs`, which is a quasi‑Newton method, not gradient descent, but it optimizes the same objective from the same standardized inputs.  
- **Same inputs.** All three use the same standardized features and the same train/validation/test split.

### Hyperparameters

I tuned only two hyperparameters on the 2022 validation season:

- **Regularization (C):** tested values from 0.001 to 100. Validation log loss was essentially flat from \(C = 1\) upwards (differences under \(5 \times 10^{-5}\)), so I chose **C = 1.0**, the strongest regularization within noise.  
- **Learning rate (PyTorch):** tested {0.1, 0.5, 1, 2}. All four reached the same validation loss, so training converged for each. I chose **0.1** as the most cautious option.

In short, with 4 features and 5,868 training games, regularization and learning rate barely mattered because the validation curve was flat.

### Test results (815 games)

| Model | Acc@0.5 | Log loss | AUC | Cost@0.75 | False pos. | False neg. | Games rested |
|--------|---------|----------|-----|-----------|------------|------------|--------------|
| scikit‑learn | 0.6319 | 0.6386 | 0.6896 | **418** | 19 | 361 | 100 |
| PyTorch manual | 0.6319 | 0.6386 | 0.6896 | **418** | 19 | 361 | 100 |
| PyTorch nn.Module | 0.6319 | 0.6386 | 0.6896 | **418** | 19 | 361 | 100 |
| Baseline (always home win) | 0.5423 | 0.6905 (train win rate) | – | **1,119** (always rest) | – | – | 815 |

## Agreement between the three implementations

The three implementations **fully agree**:

- They learn the same coefficients to four or five decimals (on standardized features; order is `pd5_diff`, `pd16_diff`, `rest_diff`, `div_game`, bias):  
  **0.2719, 0.4715, 0.0446, −0.0614, 0.2857**.  
- The largest difference in predicted probability between any two methods on the test set is about **\(1.5 \times 10^{-6}\)**, which comes from float32 (PyTorch) vs float64 (scikit‑learn) and different stopping rules (`lbfgs` vs 3,000 gradient steps).

This agreement is expected: logistic regression with an L2 penalty is a convex problem, so there is a single optimum. By matching the objective (via \(\lambda = 1/(C \cdot n)\)), using the same standardized inputs, and starting the PyTorch models from the same point, all three methods converge to the same solution. This shows that the math and implementation are correct, even if the model itself is relatively simple.

## Interpretation

- The model clearly beats the naive baseline: **63.2% vs 54.2% accuracy** and **0.6386 vs 0.6905 log loss**.  
- Coefficients match football intuition:  
  - `pd16_diff` (long‑term strength) is most important.  
  - `pd5_diff` (recent form) also matters.  
  - `rest_diff` and `div_game` have near‑zero weights, as their low correlations suggested.  
- Business impact is **small but positive**:  
  - Model cost at threshold 0.75: **418**  
  - “Never rest starters” cost: **442**  
  → about **5% reduction** in total cost.  
- The model recommends resting starters in **100 of 815 games**, and in 19 of those the team loses.

## Limitations & Next Steps

1. **Small and not fully reliable business gain.**  
   At threshold 0.75, the model saves 24 units of cost compared to “never rest”. A bootstrap over the 815 test games gives a 95% interval of about −9 to +54, so I cannot claim the model beats the simple policy with full confidence. Next step: evaluate on more seasons and report intervals, not just point estimates.

2. **Costs (3 and 1) are invented.**  
   The threshold 0.75 follows directly from these made‑up costs. If the real cost of a lost game were, say, 5 times a fatigue risk, the threshold would be 0.83. Next step: ask a team analyst for realistic cost estimates.

3. **Model overestimates home advantage in recent seasons.**  
   It predicts an average home win rate of 56.9% on the test set, while the observed rate is 54.2%. The intercept was learned mostly from older seasons (train home win rate 56.4%). Among the 100 games where it recommends resting, the home team did win 81%, so the decision area is reasonably calibrated, but the overall bias is visible. Next step: weight recent seasons more or recalibrate the intercept on the most recent seasons.

4. **Features ignore opponent strength and who is playing.**  
   A margin of +10 against weak teams counts the same as +10 against strong ones, and the dataset includes starting quarterbacks (`home_qb_id`) that I did not use, even though a quarterback change is a major shift in team strength. Next step: opponent‑adjusted margins (or an Elo‑style rating) and a “starting quarterback changed” flag.

5. **One validation season is a noisy basis for tuning.**  
   The 2022 validation season has 269 games, and the validation loss was flat for every regularization strength above \(C = 1\) and for every learning rate I tried. Tuning therefore decided almost nothing. Next step: rolling‑origin cross‑validation across several seasons.

6. **Linear model cannot capture interactions.**  
   A linear model cannot capture effects like rest mattering more when travelling. `rest_diff` and `div_game` had near‑zero weights, and I kept them for football logic rather than strong evidence. Next step: drop them or test interactions, and compare with a tree‑based model.

7. **Decision scenario is stylized.**  
   Real rest decisions depend on standings, injuries, and playoff seeding, and are made player by player, not game by game. The model estimates a win probability; turning that into a rotation policy needs football input I do not have.

8. **Test set is not perfectly clean.**  
   While exploring features, I printed test‑set numbers as well as validation ones. I chose features using validation results and simplicity, but a fully strict workflow would have hidden the test set until the end.


## Generative AI use disclosure
I used Claude (Anthropic) as an assistant only for allowed purposes in this assignment: to compare candidate datasets and help me decide on the NFL games data, to draft boilerplate code in train.py and app.py, to assist with GitHub and Visual Studio steps, and to suggest wording and formatting improvements for this report. All substantive decisions have been made on my own. I ran all code locally in Visual Studio Code and verified the outputs myself.
