"""Interactive champion draft predictor.

Loads a trained model and lets user input 5 blue + 5 red champions to predict match outcome.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd


def load_champion_names(champion_csv: str = "ChampionTbl.csv") -> Dict[int, str]:
    """Load champion ID to name mapping from CSV with friendly name aliases.
    
    Args:
        champion_csv: Path to ChampionTbl.csv
    
    Returns:
        Dictionary mapping champion ID to champion name (with aliases applied)
    """
    try:
        df = pd.read_csv(champion_csv)
        champ_map = dict(zip(df["ChampionId"], df["ChampionName"]))
        
        # Add friendly name aliases for champions with spaces or alternate names
        name_aliases = {
            "MonkeyKing": "Wukong",
            "TwistedFate": "Twisted Fate",
            "XinZhao": "Xin Zhao",
            "LeeSin": "Lee Sin",
            "JarvanIV": "Jarvan IV",
            "MissFortune": "Miss Fortune",
            "MasterYi": "Master Yi",
            "AurelionSol": "Aurelion Sol",
            "DrMundo": "Dr. Mundo",
            "TahmKench": "Tahm Kench",
            "KSante": "K'Sante",
            "RekSai": "Rek'Sai",
            "KhaZix": "Kha'Zix",
            "VelKoz": "Vel'Koz",
            "ChoGath": "Cho'Gath",
            "KogMaw": "Kog'Maw",
            "RenGar": "Rengar",
        }
        
        # Apply aliases to display names
        for champ_id, name in champ_map.items():
            if name in name_aliases:
                champ_map[champ_id] = name_aliases[name]
        
        return champ_map
    except Exception as e:
        print(f"Warning: Could not load champion names: {e}")
        return {}


def get_available_champions(model_path: Path):
    """Extract list of valid champion IDs from the trained model's feature names."""
    try:
        model_data = joblib.load(model_path)
        pipeline = model_data["pipeline"]
        
        # Extract champion columns from the preprocessor output feature names
        # ColumnTransformer format: "remainder__b_XXX" or "remainder__r_XXX"
        feat_names = pipeline.named_steps["preproc"].get_feature_names_out()
        
        champions = set()
        
        for feat in feat_names:
            # Look for "remainder__b_" or "remainder__r_" patterns
            if "__b_" in feat or "__r_" in feat:
                # Extract the number after b_ or r_
                parts = feat.replace("remainder__b_", "").replace("remainder__r_", "")
                try:
                    champ_id = int(parts)
                    champions.add(champ_id)
                except ValueError:
                    continue
        
        all_champs = sorted(champions)
        if not all_champs:
            raise ValueError("No champion features found in model. Model may need retraining.")
        
        return all_champs
    except Exception as e:
        raise ValueError(f"Failed to extract champions from model: {e}")


def input_champions(team_name: str, available: list[int], banned: set[int] | None = None, 
                    count: int = 5, champ_names: Dict[int, str] | None = None):
    """Interactively input champion IDs or names for a team with role-based selection.
    
    Args:
        team_name: Name of team (e.g., "BLUE" or "RED")
        available: List of all valid champion IDs
        banned: Set of champion IDs already picked by other team
        count: Number of champions to pick (default 5)
        champ_names: Dictionary mapping champion ID to name (optional)
    
    Returns:
        List of selected champion IDs
    """
    picks = []
    banned = banned or set()
    champ_names = champ_names or {}
    
    # Create reverse mapping: name (lowercase) -> ID
    name_to_id = {name.lower(): cid for cid, name in champ_names.items()}
    
    # Add common alternate names/aliases for easier input
    name_aliases = {
        "wukong": 62,
        "monkey": 62,
        "tf": 4,
        "xin": 5,
        "lee": 64,
        "j4": 59,
        "jarvan": 59,
        "mf": 21,
        "yi": 11,
        "asol": 136,
        "mundo": 36,
        "tahm": 223,
        "reksai": 421,
        "khazix": 121,
        "velkoz": 161,
        "chogath": 31,
        "kogmaw": 96,
        "kog": 96,
    }
    
    # Merge aliases into name_to_id
    for alias, champ_id in name_aliases.items():
        if champ_id in [cid for cid in champ_names.keys()]:  # Only if champion exists in model
            name_to_id[alias] = champ_id
    
    roles = ["Top", "Jungle", "Mid", "Bottom", "Support"]
    
    print(f"\n{team_name} Team - Select {count} champions by role")
    print(f"Hint: Type champion ID (e.g., 86) or name (e.g., Garen) or 'list' to see all champions")
    
    while len(picks) < count:
        try:
            role = roles[len(picks)]
            user_input = input(f"  {role}: ").strip()
            
            if user_input.lower() == 'list':
                print(f"\n  Available champions (showing first 30 IDs):")
                for i, cid in enumerate(available[:30]):
                    name = champ_names.get(cid, f"ID:{cid}")
                    print(f"    {cid}: {name}", end="  ")
                    if (i + 1) % 3 == 0:
                        print()
                print(f"\n  ... and {len(available) - 30} more. Total: {len(available)} champions")
                continue
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting...")
                sys.exit(0)
            
            # Try to parse as integer ID first
            champ_id = None
            try:
                champ_id = int(user_input)
            except ValueError:
                # Not a number, try to match by name
                user_input_lower = user_input.lower()
                
                # Exact match
                if user_input_lower in name_to_id:
                    champ_id = name_to_id[user_input_lower]
                else:
                    # Partial match (case-insensitive)
                    matches = [(name, cid) for name, cid in name_to_id.items() 
                              if user_input_lower in name]
                    
                    if len(matches) == 1:
                        champ_id = matches[0][1]
                        print(f"  → Matched: {champ_names.get(champ_id, champ_id)}")
                    elif len(matches) > 1:
                        print(f"  ✗ Multiple champions match '{user_input}':")
                        for name, cid in sorted(matches[:10]):
                            print(f"      - {champ_names.get(cid, cid)} (ID: {cid})")
                        if len(matches) > 10:
                            print(f"      ... and {len(matches) - 10} more")
                        print("  Please be more specific or use the champion ID")
                        continue
                    else:
                        print(f"  ✗ No champion found matching '{user_input}'")
                        print("  Tip: Try 'list' to see all champions or use champion ID")
                        continue
            
            if champ_id is None:
                print("  ✗ Invalid input. Enter a champion ID or name.")
                continue
            
            if champ_id not in available:
                print(f"  ✗ Invalid champion ID. Type 'list' to see valid IDs.")
                continue
            
            if champ_id in banned:
                print(f"  ✗ Champion {champ_id} already picked by opponent.")
                continue
            
            if champ_id in picks:
                print(f"  ✗ Champion {champ_id} already picked for this team.")
                continue
            
            picks.append(champ_id)
            champ_name = champ_names.get(champ_id, "Unknown")
            print(f"  ✓ Added champion {champ_id} ({champ_name})")
            
        except ValueError:
            print("  ✗ Invalid input. Enter a numeric champion ID.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)
    
    return picks


def input_pregame_data(team_name: str, roles: list[str]) -> Dict[str, Dict[str, float]]:
    """Collect pre-game data for each player (optional for higher confidence).
    
    Args:
        team_name: Name of team (e.g., "BLUE" or "RED")
        roles: List of role names corresponding to champion picks
    
    Returns:
        Dictionary mapping role to player stats (rank_score, mastery, win_rate)
    """
    print(f"\n{team_name} Team - Pre-Game Data (Optional - Press Enter to skip)")
    print("Tip: Adding player data increases prediction confidence")
    
    player_data = {}
    
    for role in roles:
        print(f"\n  {role} Player:")
        
        # Rank (converted to numeric score)
        while True:
            rank_input = input("    Rank (Iron/Bronze/Silver/Gold/Plat/Diamond/Master/GM/Chall) [Enter to skip]: ").strip().lower()
            if not rank_input:
                rank_score = 5.0  # Default: Gold equivalent
                break
            
            rank_map = {
                "iron": 1, "bronze": 2, "silver": 3, "gold": 5, "plat": 6, "platinum": 6,
                "diamond": 7, "master": 8, "grandmaster": 9, "gm": 9, "challenger": 10, "chall": 10
            }
            
            if rank_input in rank_map:
                rank_score = float(rank_map[rank_input])
                break
            else:
                print("      ✗ Invalid rank. Try: Iron, Bronze, Silver, Gold, Plat, Diamond, Master, GM, Chall")
        
        # Champion Mastery Points (0-1M+)
        while True:
            mastery_input = input("    Champion Mastery Points (0-1000000) [Enter to skip]: ").strip()
            if not mastery_input:
                mastery = 50000.0  # Default: moderate experience
                break
            try:
                mastery = float(mastery_input)
                if 0 <= mastery <= 10000000:
                    break
                else:
                    print("      ✗ Enter a value between 0 and 10000000")
            except ValueError:
                print("      ✗ Enter a numeric value")
        
        # Recent Win Rate (0-100%)
        while True:
            winrate_input = input("    Recent Win Rate % (0-100) [Enter to skip]: ").strip()
            if not winrate_input:
                winrate = 50.0  # Default: 50% win rate
                break
            try:
                winrate = float(winrate_input)
                if 0 <= winrate <= 100:
                    break
                else:
                    print("      ✗ Enter a value between 0 and 100")
            except ValueError:
                print("      ✗ Enter a numeric value")
        
        player_data[role] = {
            "rank_score": rank_score,
            "mastery": mastery,
            "win_rate": winrate / 100.0  # Convert to 0-1 scale
        }
    
    return player_data


def build_feature_vector(blue_champs: list[int], red_champs: list[int], 
                         pipeline, all_champions: list[int],
                         blue_pregame: Dict[str, Dict[str, float]] | None = None,
                         red_pregame: Dict[str, Dict[str, float]] | None = None):
    """Build feature vector for prediction (champions only, no player stats).
    
    Args:
        blue_champs: List of 5 blue team champion IDs
        red_champs: List of 5 red team champion IDs
        pipeline: Trained sklearn pipeline
        all_champions: List of all champion IDs seen during training
    
    Returns:
        DataFrame with properly formatted feature vector (pre-transform format)
    """
    # Build the input in the ORIGINAL format (before ColumnTransformer)
    # This should match the training data format
    
    # Calculate estimated stats from pre-game data if available
    blue_skill = 1.0
    red_skill = 1.0
    
    if blue_pregame:
        # Average rank score, mastery influence, win rate
        blue_rank_avg = np.mean([p["rank_score"] for p in blue_pregame.values()])
        blue_mastery_avg = np.mean([p["mastery"] for p in blue_pregame.values()])
        blue_winrate_avg = np.mean([p["win_rate"] for p in blue_pregame.values()])
        blue_skill = (blue_rank_avg / 5.0) * (1 + blue_mastery_avg / 200000) * (blue_winrate_avg + 0.5)
    
    if red_pregame:
        red_rank_avg = np.mean([p["rank_score"] for p in red_pregame.values()])
        red_mastery_avg = np.mean([p["mastery"] for p in red_pregame.values()])
        red_winrate_avg = np.mean([p["win_rate"] for p in red_pregame.values()])
        red_skill = (red_rank_avg / 5.0) * (1 + red_mastery_avg / 200000) * (red_winrate_avg + 0.5)
    
    # Estimate team performance based on skill differential
    skill_ratio = blue_skill / max(red_skill, 0.1)
    
    # Create base numeric columns (team stats) - estimate from pre-game data
    data = {
        "BlueBaronKills": 0.0,
        "BlueRiftHeraldKills": 0.0,
        "BlueDragonKills": 0.0,
        "BlueTowerKills": max(0, (skill_ratio - 1.0) * 3),  # Estimated based on skill
        "BlueKills": max(0, (skill_ratio - 1.0) * 10),
        "RedBaronKills": 0.0,
        "RedRiftHeraldKills": 0.0,
        "RedDragonKills": 0.0,
        "RedTowerKills": max(0, (1.0 / skill_ratio - 1.0) * 3),
        "RedKills": max(0, (1.0 / skill_ratio - 1.0) * 10),
    }
    
    # Add blue champion one-hot columns
    for champ_id in all_champions:
        data[f"b_{champ_id}"] = 1.0 if champ_id in blue_champs else 0.0
    
    # Add red champion one-hot columns
    for champ_id in all_champions:
        data[f"r_{champ_id}"] = 1.0 if champ_id in red_champs else 0.0
    
    # Add player stats columns - estimate from pre-game data if available
    # Base expected values (scaled by skill)
    blue_base_cs = 150 * blue_skill
    red_base_cs = 150 * red_skill
    blue_base_gold = 10000 * blue_skill
    red_base_gold = 10000 * red_skill
    blue_base_dmg = 15000 * blue_skill
    red_base_dmg = 15000 * red_skill
    
    data.update({
        "blue_MinionsKilled_mean": blue_base_cs,
        "blue_MinionsKilled_sum": blue_base_cs * 5,
        "red_MinionsKilled_mean": red_base_cs,
        "red_MinionsKilled_sum": red_base_cs * 5,
        "blue_DmgDealt_mean": blue_base_dmg,
        "blue_DmgDealt_sum": blue_base_dmg * 5,
        "red_DmgDealt_mean": red_base_dmg,
        "red_DmgDealt_sum": red_base_dmg * 5,
        "blue_DmgTaken_mean": red_base_dmg * 0.8,
        "blue_DmgTaken_sum": red_base_dmg * 4,
        "red_DmgTaken_mean": blue_base_dmg * 0.8,
        "red_DmgTaken_sum": blue_base_dmg * 4,
        "blue_TurretDmgDealt_mean": 2000 * blue_skill,
        "blue_TurretDmgDealt_sum": 10000 * blue_skill,
        "red_TurretDmgDealt_mean": 2000 * red_skill,
        "red_TurretDmgDealt_sum": 10000 * red_skill,
        "blue_TotalGold_mean": blue_base_gold,
        "blue_TotalGold_sum": blue_base_gold * 5,
        "red_TotalGold_mean": red_base_gold,
        "red_TotalGold_sum": red_base_gold * 5,
        "blue_kills_mean": data["BlueKills"] / 5 if data["BlueKills"] > 0 else 1.0,
        "blue_kills_sum": data["BlueKills"] if data["BlueKills"] > 0 else 5.0,
        "red_kills_mean": data["RedKills"] / 5 if data["RedKills"] > 0 else 1.0,
        "red_kills_sum": data["RedKills"] if data["RedKills"] > 0 else 5.0,
        "blue_deaths_mean": data["RedKills"] / 5 if data["RedKills"] > 0 else 1.0,
        "red_deaths_mean": data["BlueKills"] / 5 if data["BlueKills"] > 0 else 1.0,
        "blue_assists_mean": data["BlueKills"] * 0.3,
        "blue_assists_sum": data["BlueKills"] * 1.5,
        "red_assists_mean": data["RedKills"] * 0.3,
        "red_assists_sum": data["RedKills"] * 1.5,
        "blue_visionScore_mean": 30 * blue_skill,
        "blue_visionScore_sum": 150 * blue_skill,
        "red_visionScore_mean": 30 * red_skill,
        "red_visionScore_sum": 150 * red_skill,
    })
    
    return pd.DataFrame([data])


def predict_winner(blue_champs: list[int], red_champs: list[int], model_path: Path,
                  blue_pregame: Dict[str, Dict[str, float]] | None = None,
                  red_pregame: Dict[str, Dict[str, float]] | None = None):
    """Predict match outcome given team compositions and optional pre-game data.
    
    Args:
        blue_champs: List of 5 blue team champion IDs
        red_champs: List of 5 red team champion IDs
        model_path: Path to trained model joblib file
        blue_pregame: Optional pre-game data for blue team
        red_pregame: Optional pre-game data for red team
    
    Returns:
        Dict with prediction results and probabilities
    """
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    
    # Get all champions from model
    all_champions = get_available_champions(model_path)
    
    # Build feature vector with pre-game data if available
    X = build_feature_vector(blue_champs, red_champs, pipeline, all_champions, blue_pregame, red_pregame)
    
    # Predict
    pred_class = int(pipeline.predict(X)[0])
    pred_proba = pipeline.predict_proba(X)[0]
    
    blue_win_prob = float(pred_proba[1])  # class 1 = BlueWin
    red_win_prob = float(pred_proba[0])   # class 0 = RedWin
    
    return {
        "blue_win_prob": blue_win_prob,
        "red_win_prob": red_win_prob,
        "predicted_winner": "blue" if pred_class == 1 else "red",
        "prediction": "Blue" if pred_class == 1 else "Red",
        "confidence": max(blue_win_prob, red_win_prob),
    }


def main():
    ap = argparse.ArgumentParser(description="Champion draft predictor")
    ap.add_argument("--model", default="model_pipeline.joblib", help="Path to trained model")
    ap.add_argument("--champions", default="ChampionTbl.csv", help="Path to champion names CSV")
    ap.add_argument("--visualize", action="store_true", help="Generate prediction visualizations")
    ap.add_argument("--viz-output", default="prediction_viz", help="Output directory for visualizations")
    args = ap.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Train first: python train.py --out {model_path}")
        return
    
    print("=" * 60)
    print("League of Legends Match Outcome Predictor")
    print("=" * 60)
    
    # Load champion names
    champ_names = load_champion_names(args.champions)
    
    # Get available champions
    try:
        available_champs = get_available_champions(model_path)
        print(f"\nLoaded {len(available_champs)} champion types from model")
    except Exception as e:
        print(f"Error loading champions: {e}")
        return
    
    # Input phase
    print("\n" + "=" * 60)
    print("DRAFT PHASE")
    print("=" * 60)
    print("Enter champion IDs for each team (type 'list' to see all, 'quit' to exit)")
    
    roles = ["Top", "Jungle", "Mid", "Bottom", "Support"]
    blue_picks = input_champions("BLUE", available_champs, champ_names=champ_names)
    red_picks = input_champions("RED", available_champs, banned=set(blue_picks), champ_names=champ_names)
    
    # Ask if user wants to add pre-game data
    print("\n" + "="*60)
    print("PRE-GAME DATA (Optional)")
    print("="*60)
    print("Adding player data (rank, mastery, win rate) increases prediction confidence.")
    add_pregame = input("Would you like to add pre-game player data? (y/n) [n]: ").strip().lower()
    
    blue_pregame = None
    red_pregame = None
    
    if add_pregame == 'y':
        blue_pregame = input_pregame_data("BLUE", roles)
        red_pregame = input_pregame_data("RED", roles)
    
    # Display team compositions with champion names
    print("\n" + "=" * 60)
    print("TEAM COMPOSITIONS")
    print("=" * 60)
    
    print("\nBlue Team:")
    for i, champ_id in enumerate(blue_picks, 1):
        champ_name = champ_names.get(champ_id, "Unknown")
        print(f"  {i}. {champ_name:<20} (ID: {champ_id})")
    
    print("\nRed Team:")
    for i, champ_id in enumerate(red_picks, 1):
        champ_name = champ_names.get(champ_id, "Unknown")
        print(f"  {i}. {champ_name:<20} (ID: {champ_id})")
    
    # Prediction phase
    print("\n" + "=" * 60)
    print("MATCH PREDICTION")
    print("=" * 60)
    
    try:
        result = predict_winner(blue_picks, red_picks, model_path, blue_pregame, red_pregame)
        
        print(f"\n{'─' * 40}")
        print(f"  Predicted Winner: {result['prediction'].upper()} TEAM")
        print(f"{'─' * 40}")
        print(f"  Blue Win Probability: {result['blue_win_prob']:>6.1%}")
        print(f"  Red Win Probability:  {result['red_win_prob']:>6.1%}")
        print(f"{'─' * 40}")
        print(f"  Confidence Level: {result['confidence']:.1%}")
        print(f"{'─' * 40}\n")
        
    except Exception as e:
        print(f"\n✗ Prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate visualizations if requested
    if args.visualize:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        try:
            import subprocess
            cmd = [
                "python", "predict_visualize.py",
                "--model", str(args.model),
                "--champions", args.champions,
                "--blue"] + [str(c) for c in blue_picks] + [
                "--red"] + [str(c) for c in red_picks] + [
                "--output", args.viz_output
            ]
            subprocess.run(cmd, check=True)
            print(f"\n✅ Visualization opened automatically!")
            print(f"📁 All files saved to: {Path(args.viz_output).absolute()}")
        except Exception as e:
            print(f"\n⚠ Could not generate visualizations: {e}")


if __name__ == "__main__":
    main()
