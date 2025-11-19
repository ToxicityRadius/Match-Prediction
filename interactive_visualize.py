"""
Quick visualization generator - Interactive version.

Prompts user for champion IDs and generates prediction visualizations.
"""

from pathlib import Path
import subprocess
import sys

from predict import load_champion_names, get_available_champions, input_champions


def main():
    print("="*70)
    print("Interactive Prediction Visualization Generator")
    print("="*70)
    
    model_path = Path("model_test.joblib")
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("Please train a model first with: python train.py")
        sys.exit(1)
    
    # Load champion data
    champ_names = load_champion_names("ChampionTbl.csv")
    available_champs = get_available_champions(model_path)
    
    print(f"\n✓ Loaded {len(available_champs)} champions from model")
    print(f"✓ Loaded {len(champ_names)} champion names")
    
    # Get team compositions
    print("\n" + "="*70)
    print("SELECT TEAM COMPOSITIONS")
    print("="*70)
    print("Type champion IDs (e.g., 86) or names (e.g., Garen) or 'list' to see all champions\n")
    
    blue_picks = input_champions("BLUE", available_champs, champ_names=champ_names)
    red_picks = input_champions("RED", available_champs, banned=set(blue_picks), champ_names=champ_names)
    
    # Display selections
    print("\n" + "="*70)
    print("SELECTED TEAMS")
    print("="*70)
    
    print("\nBlue Team:")
    for i, cid in enumerate(blue_picks, 1):
        print(f"  {i}. {champ_names.get(cid, 'Unknown')} (ID: {cid})")
    
    print("\nRed Team:")
    for i, cid in enumerate(red_picks, 1):
        print(f"  {i}. {champ_names.get(cid, 'Unknown')} (ID: {cid})")
    
    # Ask for output directory
    print("\n" + "="*70)
    output = input("Output directory name (default: prediction_viz): ").strip()
    if not output:
        output = "prediction_viz"
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    cmd = [
        "python", "predict_visualize.py",
        "--model", str(model_path),
        "--champions", "ChampionTbl.csv",
        "--blue"] + [str(c) for c in blue_picks] + [
        "--red"] + [str(c) for c in red_picks] + [
        "--output", output
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"\n✅ Success! The main visualization has been opened automatically.")
        print(f"📁 All files saved to: {Path(output).absolute()}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating visualizations:")
        print(e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting...")
        sys.exit(0)
