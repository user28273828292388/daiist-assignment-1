"""
Assignment 1 — training pipeline (PART 1: data loading, target, leakage, split).

Business question: will the HOME team win this regular-season NFL game?
Decision it supports: if the model is confident the home team wins, the coach
rests starters. Costs (invented, see REPORT.md):
    false positive  (we say "win", rest starters, and LOSE)      = 3
    false negative  (we say "no win", play starters, would have won) = 1
=> predict "win" only when P(win) > COST_FP / (COST_FP + COST_FN) = 0.75
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "nfl_games.csv"

# --- Business framing, written as constants so every later step uses them ---
COST_FP = 3.0   # rest starters, then lose
COST_FN = 1.0   # keep starters in, would have won anyway
DECISION_THRESHOLD = COST_FP / (COST_FP + COST_FN)  # 0.75

# --- Temporal split: never train on the future ---
TRAIN_END = 2021          # train: 1999-2021
VAL_SEASON = 2022         # validation: 2022 (tune regularization / learning rate here)
TEST_START = 2023         # test: 2023-2025 (touched only for the final comparison)

# These columns describe the OUTCOME of the game (or the market's opinion of it),
# so using them to predict the outcome would be leakage.
LEAKAGE_COLS = [
    "home_score", "away_score", "result", "total", "overtime",   # the result itself
    "spread_line", "total_line", "home_moneyline", "away_moneyline",
    "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",  # market opinion
]


def load_games() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["gameday"])
    df = df[df["game_type"] == "REG"]          # playoffs have a different dynamic
    df = df[df["result"] != 0]                 # drop ties: the target must be binary
    df["home_win"] = (df["result"] > 0).astype(int)
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame):
    train = df[df["season"] <= TRAIN_END]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] >= TEST_START]
    return train, val, test


def main() -> None:
    games = load_games()
    train, val, test = temporal_split(games)

    print(f"Regular-season games (no ties): {len(games)}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(
            f"{name:>5}: {len(part):5d} games | seasons {part['season'].min()}-{part['season'].max()}"
            f" | home win rate {part['home_win'].mean():.3f}"
        )
    print(f"Decision threshold from costs: {DECISION_THRESHOLD:.2f}")

    # Naive baseline: always predict "home wins"
    baseline_acc = (test["home_win"] == 1).mean()
    print(f"Naive baseline (always home win) test accuracy: {baseline_acc:.3f}")


if __name__ == "__main__":
    main()
