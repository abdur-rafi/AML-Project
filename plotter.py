import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(csv_file, y_range=None, save_path=None):
    # Read the CSV
    df = pd.read_csv(csv_file)

    # Set seaborn style for beauty
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(10, 6))
    x = range(1, len(df["Accuracy"]) + 1)
    plt.plot(x, df["Accuracy"], label="Accuracy", marker="o", linewidth=2, color="#1f77b4")
    plt.plot(x, df["F1"], label="F1 Score", marker="s", linewidth=2, color="#ff7f0e")

    # Highlight max Accuracy
    max_acc_idx = df["Accuracy"].idxmax()
    max_acc_x = max_acc_idx + 1
    max_acc_val = df["Accuracy"].iloc[max_acc_idx]
    plt.scatter(max_acc_x, max_acc_val, color="#2ca02c", s=60, zorder=5, label=f"Max Accuracy: {max_acc_val:.2f} (Gen {max_acc_x})")
    plt.annotate(f"{max_acc_val:.2f}", (max_acc_x, max_acc_val), textcoords="offset points", xytext=(0,10), ha='center', color="#2ca02c", fontsize=12, fontweight="bold")

    # Highlight max F1
    max_f1_idx = df["F1"].idxmax()
    max_f1_x = max_f1_idx + 1
    max_f1_val = df["F1"].iloc[max_f1_idx]
    plt.scatter(max_f1_x, max_f1_val, color="#d62728", s=60, zorder=5, label=f"Max F1: {max_f1_val:.2f} (Gen {max_f1_x})")
    plt.annotate(f"{max_f1_val:.2f}", (max_f1_x, max_f1_val), textcoords="offset points", xytext=(0,-15), ha='center', color="#d62728", fontsize=12, fontweight="bold")

    plt.title("Accuracy and F1 Score Over Generations", fontsize=18, fontweight="bold")
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Score (%)", fontsize=14)
    plt.legend(loc="best", fontsize=13)
    plt.tight_layout()

    # Optionally set y-axis range
    if y_range is not None:
        plt.ylim(y_range)

    # Add grid and minor ticks for clarity
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    # Usage: python plot_metrics.py <csv_file> [ymin] [ymax] [save_path]
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <csv_file> [ymin] [ymax] [save_path]")
        sys.exit(1)
    csv_file = sys.argv[1]
    y_range = None
    save_path = None
    # if len(sys.argv) >= 4:
    #     y_range = (float(sys.argv[2]), float(sys.argv[3]))
    if len(sys.argv) == 3:
        save_path = sys.argv[2]
    plot_metrics(csv_file, y_range, save_path)