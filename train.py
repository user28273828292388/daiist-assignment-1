"""
Assignment 1 — training pipeline (PART 2: features, leakage check, scaled arrays).

Business question: will the HOME team win this regular-season NFL game?
Decision it supports: if the model is confident the home team wins, the coach
rests starters. Costs (invented, see REPORT.md):
    false positive  (we say "win", rest starters, and LOSE)         = 3
    false negative  (we say "no win", play starters, would have won) = 1
=> predict "win" only when P(win) > COST_FP / (COST_FP + COST_FN) = 0.75
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# Final feature set. All are "home minus away" so one number says who is stronger.
FEATURES = [
    "pd5_diff",    # recent form: avg point margin in the last 5 games
    "pd16_diff",   # underlying strength: avg point margin in the last 16 games
    "rest_diff",   # days of rest, home minus away
    "div_game",    # 1 if both teams are in the same division
]
SHORT_WINDOW, LONG_WINDOW = 5, 16


# ----------------------------------------------------------------------------
# 1. Loading and target
# ----------------------------------------------------------------------------
def load_games() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["gameday"])
    df = df[df["game_type"] == "REG"]          # playoffs have a different dynamic
    df = df[df["result"] != 0]                 # drop ties: the target must be binary
    df["home_win"] = (df["result"] > 0).astype(int)
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Features built ONLY from games played before the one we predict
# ----------------------------------------------------------------------------
def team_form(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with that team's average point margin over its
    previous SHORT_WINDOW / LONG_WINDOW games. `shift(1)` is the key line: it
    removes the current game, so the feature only sees the past."""
    home = games[["game_id", "gameday", "home_team", "home_score", "away_score"]]
    away = games[["game_id", "gameday", "away_team", "away_score", "home_score"]]
    cols = ["game_id", "gameday", "team", "points_for", "points_against"]
    long = pd.concat([home.set_axis(cols, axis=1), away.set_axis(cols, axis=1)])
    long = long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    long["margin"] = long["points_for"] - long["points_against"]

    by_team = long.groupby("team")["margin"]
    for window in (SHORT_WINDOW, LONG_WINDOW):
        long[f"pd{window}"] = by_team.transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    # A team's very first game in the data (1999 week 1) has no past: neutral value 0.
    return long[["game_id", "team", f"pd{SHORT_WINDOW}", f"pd{LONG_WINDOW}"]].fillna(0.0)


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    form = team_form(games)
    out = games.copy()
    for side in ("home", "away"):
        side_form = form.rename(
            columns={
                "team": f"{side}_team",
                f"pd{SHORT_WINDOW}": f"{side}_pd5",
                f"pd{LONG_WINDOW}": f"{side}_pd16",
            }
        )
        out = out.merge(side_form, on=["game_id", f"{side}_team"], how="left")
    out["pd5_diff"] = out["home_pd5"] - out["away_pd5"]
    out["pd16_diff"] = out["home_pd16"] - out["away_pd16"]
    out["rest_diff"] = out["home_rest"] - out["away_rest"]
    return out


def check_no_future_leakage(games: pd.DataFrame, features: pd.DataFrame) -> None:
    """Sanity test. Pick the first game day of the test period and scramble the
    scores of every game on or after it. Features of the games ON that day must
    not move (they may only depend on earlier games), while features of later
    games must move (proving the test has teeth)."""
    cutoff = games.loc[games["season"] >= TEST_START, "gameday"].min()
    rng = np.random.default_rng(0)
    scrambled = games.copy()
    mask = scrambled["gameday"] >= cutoff
    scrambled.loc[mask, "home_score"] = rng.integers(0, 50, mask.sum())
    scrambled.loc[mask, "away_score"] = rng.integers(0, 50, mask.sum())
    other = build_features(scrambled)

    same_day = games["gameday"] == cutoff
    later = games["gameday"] > cutoff + pd.Timedelta(days=60)
    cols = ["pd5_diff", "pd16_diff"]
    assert np.allclose(features.loc[same_day, cols], other.loc[same_day, cols]), \
        "LEAKAGE: features changed when only the present/future scores changed"
    assert not np.allclose(features.loc[later, cols], other.loc[later, cols]), \
        "Leakage test is broken: scrambling the future changed nothing"
    print("Leakage check passed: features depend only on past games.")


# ----------------------------------------------------------------------------
# 3. Split + scaling (scaler fitted on TRAIN only)
# ----------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame):
    train = df[df["season"] <= TRAIN_END]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] >= TEST_START]
    return train, val, test


def prepare_data():
    games = load_games()
    features = build_features(games)
    assert not set(FEATURES) & set(LEAKAGE_COLS), "a leakage column is a feature"
    check_no_future_leakage(games, features)

    train, val, test = temporal_split(features)
    scaler = StandardScaler().fit(train[FEATURES])      # statistics from train only
    data = {"scaler": scaler, "features": FEATURES}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        data[f"X_{name}"] = scaler.transform(part[FEATURES]).astype(np.float32)
        data[f"y_{name}"] = part["home_win"].to_numpy(dtype=np.float32)
        data[f"meta_{name}"] = part[["game_id", "season", "week", "gameday",
                                     "home_team", "away_team"]].reset_index(drop=True)
    return data, (train, val, test)


def main() -> None:
    data, (train, val, test) = prepare_data()

    print(f"\nFeatures used: {FEATURES}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(
            f"{name:>5}: {len(part):5d} games | seasons {part['season'].min()}-{part['season'].max()}"
            f" | home win rate {part['home_win'].mean():.3f}"
        )
    print("\nCorrelation of each feature with home_win (train):")
    print(train[FEATURES + ["home_win"]].corr()["home_win"].drop("home_win").round(3).to_string())
    print(f"\nDecision threshold from costs: {DECISION_THRESHOLD:.2f}")
    print(f"Naive baseline (always home win) test accuracy: {(test['home_win'] == 1).mean():.3f}")


if __name__ == "__main__":
    main()
