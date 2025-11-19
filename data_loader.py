import pandas as pd
import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import MultiLabelBinarizer


def aggregate_player_stats(summoner_match_csv: str, match_stats_csv: str, team_match_df: pd.DataFrame):
    """Aggregate per-player stats to team-level features.
    
    Returns a dict mapping MatchFk -> {'blue_stats': dict, 'red_stats': dict}
    where each team dict contains aggregated KDA, gold, cs, damage, vision etc.
    """
    summoner_match = pd.read_csv(summoner_match_csv)
    match_stats = pd.read_csv(match_stats_csv)
    
    # merge to get champion + stats for each player
    merged = summoner_match.merge(match_stats, left_on="SummonerMatchId", right_on="SummonerMatchFk", how="inner")
    
    # map summoner to team using TeamMatchTbl: blue players have indices 0-4, red 5-9 in TeamMatchTbl ordering
    # infer team from Lane + MatchFk (blue vs red side)
    # simpler: use Win column; if Win=1 and BlueWin=1 or Win=0 and BlueWin=0, then blue; else red
    
    # Join with team match to get BlueWin label
    merged = merged.merge(
        team_match_df[["MatchFk", "BlueWin"]],
        left_on="MatchFk", right_on="MatchFk", how="left"
    )
    
    # Infer team: if Win == BlueWin, player is on blue team
    merged["is_blue"] = (merged["Win"] == merged["BlueWin"]).astype(int)
    
    # Aggregate stats by match and team
    agg_dict = {
        "MinionsKilled": ["mean", "sum"],
        "DmgDealt": ["mean", "sum"],
        "DmgTaken": ["mean", "sum"],
        "TurretDmgDealt": ["mean", "sum"],
        "TotalGold": ["mean", "sum"],
        "kills": ["mean", "sum"],
        "deaths": ["mean"],
        "assists": ["mean", "sum"],
        "visionScore": ["mean", "sum"],
    }
    
    result = {}
    for match_id in team_match_df["MatchFk"].unique():
        match_data = merged[merged["MatchFk"] == match_id]
        
        blue_data = match_data[match_data["is_blue"] == 1]
        red_data = match_data[match_data["is_blue"] == 0]
        
        blue_stats = {}
        red_stats = {}
        
        for col, funcs in agg_dict.items():
            if col in blue_data.columns:
                for func in funcs:
                    key = f"{col}_{func}"
                    blue_stats[key] = blue_data[col].agg(func) if len(blue_data) > 0 else 0
                    red_stats[key] = red_data[col].agg(func) if len(red_data) > 0 else 0
        
        result[match_id] = {"blue_stats": blue_stats, "red_stats": red_stats}
    
    return result


def load_teammatch_features(
    team_csv: str,
    summoner_match_csv: str | None = None,
    match_stats_csv: str | None = None,
    nrows: int | None = None,
):
    """Load TeamMatchTbl.csv and optionally aggregate player stats.

    Returns (X, y, mlb_blue, mlb_red, numeric_cols)
    - X: pandas.DataFrame with numeric + one-hot champion + player stats
    - y: pandas.Series (BlueWin: 1 = blue team won)
    - mlb_blue, mlb_red: fitted MultiLabelBinarizer for later inference
    - numeric_cols: list of numeric column names included in X
    """
    df = pd.read_csv(team_csv, nrows=nrows)

    # label: predict whether blue team won
    if "BlueWin" not in df.columns:
        raise ValueError("Expected column 'BlueWin' in TeamMatchTbl.csv")
    y = df["BlueWin"].astype(int)

    # champion columns
    blue_cols = [f"B{i}Champ" for i in range(1, 6)]
    red_cols = [f"R{i}Champ" for i in range(1, 6)]

    for c in blue_cols + red_cols:
        if c not in df.columns:
            df[c] = ""

    # convert champion ids to strings and build lists per row
    blue_lists = df[blue_cols].astype(str).values.tolist()
    red_lists = df[red_cols].astype(str).values.tolist()

    mlb_b = MultiLabelBinarizer(sparse_output=False)
    mlb_r = MultiLabelBinarizer(sparse_output=False)

    B_raw = mlb_b.fit_transform(blue_lists)
    # ensure we have a dense numpy array for pandas.DataFrame (Pylance type clarity)
    if issparse(B_raw):
        B = np.asarray(B_raw.toarray(), dtype=float)  # type: ignore[attr-defined]
    else:
        B = np.asarray(B_raw).astype(float)

    R_raw = mlb_r.fit_transform(red_lists)
    if issparse(R_raw):
        R = np.asarray(R_raw.toarray(), dtype=float)  # type: ignore[attr-defined]
    else:
        R = np.asarray(R_raw).astype(float)

    b_cols = ["b_" + str(c) for c in mlb_b.classes_]
    r_cols = ["r_" + str(c) for c in mlb_r.classes_]

    B_df = pd.DataFrame(B, columns=b_cols, index=df.index)
    R_df = pd.DataFrame(R, columns=r_cols, index=df.index)

    # numeric game/team stats that exist in the file; fall back to zeros when missing
    numeric_cols = [
        "BlueBaronKills",
        "BlueRiftHeraldKills",
        "BlueDragonKills",
        "BlueTowerKills",
        "BlueKills",
        "RedBaronKills",
        "RedRiftHeraldKills",
        "RedDragonKills",
        "RedTowerKills",
        "RedKills",
    ]

    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0

    num_df = df[numeric_cols].fillna(0).astype(float)

    # Optionally load and aggregate player stats
    player_stats_features = None
    player_stat_cols = []
    if summoner_match_csv and match_stats_csv:
        print("Aggregating player stats...")
        player_stats_dict = aggregate_player_stats(summoner_match_csv, match_stats_csv, df)
        
        # Build player stats dataframe
        player_stats_list = []
        for idx, match_id in enumerate(df["MatchFk"]):
            if match_id in player_stats_dict:
                stats = player_stats_dict[match_id]
                row = {}
                for key in stats["blue_stats"].keys():
                    row[f"blue_{key}"] = stats["blue_stats"].get(key, 0)
                    row[f"red_{key}"] = stats["red_stats"].get(key, 0)
                player_stats_list.append(row)
            else:
                player_stats_list.append({})
        
        if player_stats_list and player_stats_list[0]:
            player_stats_features = pd.DataFrame(player_stats_list, index=df.index).fillna(0)
            player_stat_cols = list(player_stats_features.columns)

    feature_dfs = [num_df, B_df, R_df]
    if player_stats_features is not None:
        feature_dfs.append(player_stats_features)
    
    X = pd.concat(feature_dfs, axis=1)
    
    return X, y, mlb_b, mlb_r, numeric_cols + player_stat_cols


if __name__ == "__main__":
    import sys

    team_csv = sys.argv[1] if len(sys.argv) > 1 else "TeamMatchTbl.csv"
    summoner_csv = sys.argv[2] if len(sys.argv) > 2 else "SummonerMatchTbl.csv"
    stats_csv = sys.argv[3] if len(sys.argv) > 3 else "MatchStatsTbl.csv"

    print("Loading without player stats...")
    X_base, y_base, _, _, cols_base = load_teammatch_features(team_csv, nrows=100)
    print(f"Base features: shape={X_base.shape}, cols={len(cols_base)}")

    print("\nLoading with player stats aggregation...")
    X_full, y_full, _, _, cols_full = load_teammatch_features(
        team_csv, summoner_csv, stats_csv, nrows=100
    )
    print(f"Full features: shape={X_full.shape}, cols={len(cols_full)}")
    print(f"Added {len(cols_full) - len(cols_base)} player stat features")
