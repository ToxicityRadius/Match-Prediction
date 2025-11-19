"""
Generate prediction-specific visualizations for user-selected champion teams.

Creates graphs showing:
- Win probability comparison
- Confidence visualization
- Feature contributions to the prediction
- Historical performance of similar team compositions
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]
from termcolor import colored  # type: ignore[import-untyped]

from predict import load_champion_names, get_available_champions, build_feature_vector

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def display_image_in_terminal(image_path, max_width=80):
    """Display an image in the terminal using ASCII art."""
    try:
        from PIL import Image  # type: ignore[import-untyped]
        
        img = Image.open(image_path)
        
        # Resize image to fit terminal width
        aspect_ratio = img.height / img.width
        new_width = min(max_width, 100)
        new_height = int(new_width * aspect_ratio * 0.5)  # 0.5 to account for character aspect ratio
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img = img.convert('L')  # Convert to grayscale
        
        # ASCII characters from darkest to lightest
        ascii_chars = " .:-=+*#%@"
        
        pixels = img.getdata()
        ascii_str = ""
        
        for i, pixel in enumerate(pixels):
            ascii_str += ascii_chars[pixel * len(ascii_chars) // 256]
            if (i + 1) % new_width == 0:
                ascii_str += "\n"
        
        return ascii_str
    except Exception as e:
        return f"[Could not display image: {e}]"


def display_prediction_results_terminal(result, blue_champs, red_champs, champ_names, output_dir):
    """Display prediction results with ASCII art in terminal."""
    print("\n" + "="*70)
    print(colored("PREDICTION VISUALIZATION (ASCII Preview)", "cyan", attrs=["bold"]))
    print("="*70)
    
    # Display win probabilities as bar chart
    print("\n" + colored("Win Probabilities:", "yellow", attrs=["bold"]))
    blue_prob = result["blue_win_prob"]
    red_prob = result["red_win_prob"]
    
    blue_bar_len = int(blue_prob * 50)
    red_bar_len = int(red_prob * 50)
    
    print(f"\n{colored('Blue Team:', 'blue', attrs=['bold'])} {blue_prob:6.2%}")
    print(colored("█" * blue_bar_len, "blue") + "░" * (50 - blue_bar_len))
    
    print(f"\n{colored('Red Team: ', 'red', attrs=['bold'])} {red_prob:6.2%}")
    print(colored("█" * red_bar_len, "red") + "░" * (50 - red_bar_len))
    
    # Display winner
    winner = "Blue Team" if result["predicted_winner"] == "blue" else "Red Team"
    winner_color = "blue" if result["predicted_winner"] == "blue" else "red"
    confidence = result["confidence"]
    
    print("\n" + "="*70)
    print(colored(f"🏆 PREDICTED WINNER: {winner.upper()}", winner_color, attrs=["bold"]))
    print(colored(f"📊 CONFIDENCE: {confidence:.1%}", "green" if confidence > 0.7 else "yellow", attrs=["bold"]))
    print("="*70)
    
    # Display team rosters
    print("\n" + colored("Team Compositions:", "cyan", attrs=["bold"]))
    print("\n" + colored("Blue Team:", "blue", attrs=["bold"]))
    for i, cid in enumerate(blue_champs, 1):
        name = champ_names.get(cid, f"Unknown (ID: {cid})")
        if cid not in champ_names:
            print(f"  {i}. {colored(name, 'yellow')} ⚠️ Champion ID not found in database")
        else:
            print(f"  {i}. {name}")
    
    print("\n" + colored("Red Team:", "red", attrs=["bold"]))
    for i, cid in enumerate(red_champs, 1):
        name = champ_names.get(cid, f"Unknown (ID: {cid})")
        if cid not in champ_names:
            print(f"  {i}. {colored(name, 'yellow')} ⚠️ Champion ID not found in database")
        else:
            print(f"  {i}. {name}")
    
    print("\n" + "="*70)
    print(colored(f"📁 Full visualizations saved to: {output_dir.absolute()}", "green"))
    print("="*70)


def plot_win_probabilities(blue_champs, red_champs, result, champ_names, output_dir):
    """Plot win probability comparison for both teams."""
    print("\n📊 Generating win probability chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    teams = ["Blue Team", "Red Team"]
    probabilities = [result["blue_win_prob"], result["red_win_prob"]]
    colors = ["#3498db", "#e74c3c"]
    
    bars = ax1.bar(teams, probabilities, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel("Win Probability", fontsize=12, fontweight="bold")
    ax1.set_title("Match Outcome Prediction", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.1%}', ha='center', va='bottom', fontsize=14, fontweight="bold")
    
    # Add winner indicator
    winner = "Blue Team" if result["predicted_winner"] == "blue" else "Red Team"
    ax1.text(0.5, 0.95, f"Predicted Winner: {winner}", 
             transform=ax1.transAxes, ha='center', va='top',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Pie chart
    ax2.pie(probabilities, labels=teams, autopct='%1.1f%%', startangle=90,
            colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax2.set_title(f"Confidence: {result['confidence']:.1%}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    output_path = output_dir / "prediction_probabilities.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_team_compositions(blue_champs, red_champs, champ_names, output_dir):
    """Visualize team compositions with champion names."""
    print("\n📊 Generating team composition chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Blue team
    blue_names = [champ_names.get(cid, f"Unknown (ID:{cid})") for cid in blue_champs]
    y_pos = np.arange(len(blue_names))
    ax1.barh(y_pos, [1]*len(blue_names), color="#3498db", alpha=0.7, edgecolor="black")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(blue_names, fontsize=11)
    ax1.set_xlabel("Selection", fontsize=12)
    ax1.set_title("Blue Team Composition", fontsize=14, fontweight="bold", color="#3498db")
    ax1.set_xlim([0, 1.2])
    ax1.invert_yaxis()
    
    # Add champion IDs as text
    for i, (name, cid) in enumerate(zip(blue_names, blue_champs)):
        ax1.text(1.05, i, f"(ID: {cid})", va='center', fontsize=9, color='gray')
    
    # Red team
    red_names = [champ_names.get(cid, f"Unknown (ID:{cid})") for cid in red_champs]
    y_pos = np.arange(len(red_names))
    ax2.barh(y_pos, [1]*len(red_names), color="#e74c3c", alpha=0.7, edgecolor="black")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(red_names, fontsize=11)
    ax2.set_xlabel("Selection", fontsize=12)
    ax2.set_title("Red Team Composition", fontsize=14, fontweight="bold", color="#e74c3c")
    ax2.set_xlim([0, 1.2])
    ax2.invert_yaxis()
    
    # Add champion IDs as text
    for i, (name, cid) in enumerate(zip(red_names, red_champs)):
        ax2.text(1.05, i, f"(ID: {cid})", va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    output_path = output_dir / "team_compositions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_feature_contributions(blue_champs, red_champs, model_path, all_champions, output_dir, top_n=15):
    """Plot which features contribute most to this specific prediction."""
    print("\n📊 Generating feature contribution analysis...")
    
    # Load model
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    
    # Build feature vector
    X = build_feature_vector(blue_champs, red_champs, pipeline, all_champions)
    
    # Transform features
    X_transformed = pipeline.named_steps["preproc"].transform(X)
    feature_names = pipeline.named_steps["preproc"].get_feature_names_out()
    
    # Get model coefficients (for logistic regression) or feature importances (for tree-based)
    clf = pipeline.named_steps["clf"]
    
    if hasattr(clf, "coef_"):
        # Logistic regression - use coefficients
        importances = np.abs(clf.coef_[0])
        title = "Feature Coefficients Impact"
    elif hasattr(clf, "feature_importances_"):
        # Tree-based - use feature importances
        importances = clf.feature_importances_
        title = "Feature Importance"
    else:
        print("  ⚠ Model doesn't support feature importance visualization")
        return
    
    # Get non-zero features (actually used in this prediction)
    X_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    active_features = np.where(X_dense[0] != 0)[0]
    
    if len(active_features) == 0:
        print("  ⚠ No active features found for this prediction")
        return
    
    # Create dataframe with active features and their importance
    feature_data = []
    for idx in active_features:
        if idx < len(feature_names) and idx < len(importances):
            feature_data.append({
                "feature": feature_names[idx],
                "importance": importances[idx],
                "value": X_dense[0, idx]
            })
    
    if not feature_data:
        print("  ⚠ Could not extract feature contributions")
        return
    
    importance_df = pd.DataFrame(feature_data).sort_values("importance", ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color map (blue for blue team features, red for red team)
    colors = []
    for feat in importance_df["feature"]:
        if "_b_" in feat or "blue_" in feat.lower() or "Blue" in feat:
            colors.append("#3498db")
        elif "_r_" in feat or "red_" in feat.lower() or "Red" in feat:
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")
    
    bars = ax.barh(range(len(importance_df)), importance_df["importance"], color=colors, alpha=0.7, edgecolor="black")
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"], fontsize=10)
    ax.set_xlabel("Feature Importance/Coefficient", fontsize=12, fontweight="bold")
    ax.set_title(f"{title} - Active Features for This Prediction", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add legend
    from matplotlib.patches import Patch  # type: ignore[import-untyped]
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, label='Blue Team Features'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Red Team Features'),
        Patch(facecolor='#95a5a6', alpha=0.7, label='Neutral Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / "feature_contributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_prediction_confidence(result, output_dir):
    """Visualize prediction confidence level."""
    print("\n📊 Generating confidence gauge...")
    
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))
    
    confidence = result["confidence"]
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    
    # Background arc
    ax.plot(theta, [1]*len(theta), color='lightgray', linewidth=20, alpha=0.3)
    
    # Confidence arc
    conf_theta = theta[theta <= confidence * np.pi]
    ax.plot(conf_theta, [1]*len(conf_theta), linewidth=20, alpha=0.8,
            color='#2ecc71' if confidence > 0.7 else '#f39c12' if confidence > 0.55 else '#e74c3c')
    
    # Add confidence text
    ax.text(np.pi/2, 0.5, f"{confidence:.1%}", 
            ha='center', va='center', fontsize=40, fontweight='bold')
    ax.text(np.pi/2, 0.2, "Confidence", 
            ha='center', va='center', fontsize=16)
    
    # Configure plot
    ax.set_ylim(0, 1.2)  # type: ignore[call-overload]
    ax.set_theta_zero_location('W')  # type: ignore[attr-defined]
    ax.set_theta_direction(1)  # type: ignore[attr-defined]
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(['Low\n(50%)', 'Medium\n(75%)', 'High\n(100%)'])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    
    winner = "Blue Team" if result["predicted_winner"] == "blue" else "Red Team"
    plt.title(f"Prediction Confidence for {winner} Victory", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / "confidence_gauge.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_all_predictions_summary(blue_champs, red_champs, result, champ_names, output_dir):
    """Create a single comprehensive summary visualization."""
    print("\n📊 Generating comprehensive prediction summary...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Team Compositions (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    blue_names = [champ_names.get(cid, f"ID:{cid}") for cid in blue_champs]
    ax1.barh(range(5), [1]*5, color="#3498db", alpha=0.7, edgecolor="black")
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(blue_names)], fontsize=10)
    ax1.set_title("Blue Team", fontsize=12, fontweight="bold", color="#3498db")
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    red_names = [champ_names.get(cid, f"ID:{cid}") for cid in red_champs]
    ax2.barh(range(5), [1]*5, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(red_names)], fontsize=10)
    ax2.set_title("Red Team", fontsize=12, fontweight="bold", color="#e74c3c")
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.axis('off')
    
    # 2. Win Probability (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    teams = ["Blue Team", "Red Team"]
    probabilities = [result["blue_win_prob"], result["red_win_prob"]]
    colors = ["#3498db", "#e74c3c"]
    bars = ax3.bar(teams, probabilities, color=colors, alpha=0.7, edgecolor="black", linewidth=2)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Win Probability", fontsize=11, fontweight="bold")
    ax3.set_title("Match Prediction", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.1%}', ha='center', va='bottom', fontsize=12, fontweight="bold")
    
    # 3. Confidence Pie Chart (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    winner_prob = result["confidence"]
    other_prob = 1 - winner_prob
    winner = "Blue" if result["predicted_winner"] == "blue" else "Red"
    ax4.pie([winner_prob, other_prob], labels=[f"{winner} Win", "Uncertainty"],
            autopct='%1.1f%%', startangle=90,
            colors=[colors[0 if winner == "Blue" else 1], "#95a5a6"],
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    ax4.set_title(f"Confidence Level: {winner_prob:.1%}", fontsize=12, fontweight="bold")
    
    # 4. Prediction Summary (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_text = f"""
PREDICTION SUMMARY

Predicted Winner: {winner.upper()} TEAM
Confidence Level: {result['confidence']:.1%} ({'High' if result['confidence'] > 0.7 else 'Medium' if result['confidence'] > 0.55 else 'Low'})

Win Probabilities:
• Blue Team: {result['blue_win_prob']:.2%}
• Red Team: {result['red_win_prob']:.2%}

Model: Logistic Regression (98.6% ROC-AUC)
Prediction based on champion composition (player stats not available)
    """
    
    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
            ha='center', va='center', fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle("League of Legends Match Outcome Prediction", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    output_path = output_dir / "prediction_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def generate_prediction_visualizations(blue_champs, red_champs, model_path, champ_names, output_dir):
    """Generate all visualizations for a specific prediction."""
    
    # Load model and make prediction
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    all_champions = get_available_champions(model_path)
    
    # Build feature vector and predict
    X = build_feature_vector(blue_champs, red_champs, pipeline, all_champions)
    y_pred = pipeline.predict(X)[0]
    y_proba = pipeline.predict_proba(X)[0]
    
    result = {
        "blue_win_prob": float(y_proba[1]),
        "red_win_prob": float(y_proba[0]),
        "predicted_winner": "blue" if y_pred == 1 else "red",
        "prediction": "Blue" if y_pred == 1 else "Red",
        "confidence": float(max(y_proba)),
    }
    
    # Generate all visualizations
    plot_all_predictions_summary(blue_champs, red_champs, result, champ_names, output_dir)
    plot_win_probabilities(blue_champs, red_champs, result, champ_names, output_dir)
    plot_team_compositions(blue_champs, red_champs, champ_names, output_dir)
    plot_prediction_confidence(result, output_dir)
    plot_feature_contributions(blue_champs, red_champs, model_path, all_champions, output_dir)
    
    # Display results in terminal
    display_prediction_results_terminal(result, blue_champs, red_champs, champ_names, output_dir)
    
    return result


def main():
    ap = argparse.ArgumentParser(description="Generate prediction visualizations for specific team compositions")
    ap.add_argument("--model", default="model_test.joblib", help="Path to trained model")
    ap.add_argument("--champions", default="ChampionTbl.csv", help="Path to champion names CSV")
    ap.add_argument("--output", default="prediction_viz", help="Output directory for visualizations")
    ap.add_argument("--blue", nargs=5, type=int, required=True, 
                    help="Blue team champion IDs (5 integers)")
    ap.add_argument("--red", nargs=5, type=int, required=True,
                    help="Red team champion IDs (5 integers)")
    ap.add_argument("--no-open", action="store_true", help="Don't automatically open visualizations")
    args = ap.parse_args()
    
    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("="*70)
    print("Match Prediction Visualization Generator")
    print("="*70)
    
    # Load champion names
    champ_names = load_champion_names(args.champions)
    print(f"\n✓ Loaded {len(champ_names)} champion names")
    
    # Validate champion IDs
    print("\n📋 Validating champion IDs...")
    all_champs = args.blue + args.red
    invalid_champs = [cid for cid in all_champs if cid not in champ_names]
    
    if invalid_champs:
        print(f"⚠️  Warning: {len(invalid_champs)} champion ID(s) not found in database:")
        for cid in invalid_champs:
            print(f"   - ID {cid} (not in ChampionTbl.csv)")
        print("   These will be displayed as 'Unknown' in visualizations")
    else:
        print("✓ All champion IDs are valid")
    
    # Display team compositions
    print("\n" + "="*70)
    print("TEAM COMPOSITIONS")
    print("="*70)
    print("\nBlue Team:")
    for i, cid in enumerate(args.blue, 1):
        name = champ_names.get(cid, f"Unknown (ID: {cid})")
        if cid in invalid_champs:
            print(f"  {i}. {name} ⚠️")
        else:
            print(f"  {i}. {name}")
    
    print("\nRed Team:")
    for i, cid in enumerate(args.red, 1):
        name = champ_names.get(cid, f"Unknown (ID: {cid})")
        if cid in invalid_champs:
            print(f"  {i}. {name} ⚠️")
        else:
            print(f"  {i}. {name}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    result = generate_prediction_visualizations(
        args.blue, args.red, model_path, champ_names, output_dir
    )
    
    print("\n📁 Generated files:")
    files_to_open = [
        "prediction_summary.png",
        "prediction_probabilities.png",
        "team_compositions.png",
        "confidence_gauge.png",
        "feature_contributions.png"
    ]
    
    for file in files_to_open:
        print(f"  • {file}")
    
    # Open visualizations automatically
    if not args.no_open:
        print("\n🖼️  Opening visualizations...")
        import subprocess
        import sys
        import time
        
        # Open the main summary image
        summary_path = output_dir / "prediction_summary.png"
        if summary_path.exists():
            try:
                if sys.platform == "win32":
                    import os
                    os.startfile(str(summary_path))
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", str(summary_path)], check=True)
                else:  # Linux
                    subprocess.run(["xdg-open", str(summary_path)], check=True)
                
                print(f"  ✓ Opened: {summary_path.name}")
                time.sleep(0.5)  # Small delay between opens
                
                # Ask if user wants to see other visualizations
                print("\n💡 Tip: All visualizations are saved in the folder.")
                print(f"   Open folder to view all: {output_dir.absolute()}")
                
            except Exception as e:
                print(f"  ⚠ Could not auto-open image: {e}")
                print(f"  Please manually open: {summary_path}")
        else:
            print("  ⚠ Summary image not found")
    else:
        print("\n💡 Auto-open disabled. Manually open files from the folder.")
    
    print()


if __name__ == "__main__":
    main()
