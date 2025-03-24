import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
sns.set(style="whitegrid")

# File & folder config
CSV_FILE = "summary_eval_scores.csv"
SAVE_DIR = "analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_FILE)

def plot_hallucination_scores(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="Testcase ID",
        y="Hallucination Score",
        hue="Model",
        palette="muted"
    )
    plt.title("Hallucination Score Comparison Across Models")
    plt.ylabel("Hallucination Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR, "hallucination_scores.png")
    plt.savefig(filename)
    print(f"📈 Saved: {filename}")
    plt.show()

def plot_metric_grouped(df, metric_names):
    for metric in metric_names:
        metric_cols = [col for col in df.columns if col.startswith(metric) and "Score" in col]
        if not metric_cols:
            continue

        melted = df.melt(
            id_vars=["Testcase ID", "Model"],
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score"
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=melted,
            x="Testcase ID",
            y="Score",
            hue="Model",
            palette="Set2"
        )
        plt.title(f"{metric} Score Comparison")
        plt.ylabel(f"{metric} Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = os.path.join(SAVE_DIR, f"{metric.lower().replace(' ', '_')}_scores.png")
        plt.savefig(filename)
        print(f"📈 Saved: {filename}")
        plt.show()

def main():
    print("📊 Generating EDA Charts...")

    # Hallucination specific
    if "Hallucination Score" in df.columns:
        plot_hallucination_scores(df)
    else:
        print("⚠️ 'Hallucination Score' column not found.")

    # General metrics
    metrics_to_plot = [
        "ROUGE Scores", "BERTScore", "Bias",
        "Toxicity", "Summarization", "Prompt Alignment", "Hallucination"
    ]
    plot_metric_grouped(df, metrics_to_plot)

    print(f"\n✅ All plots saved to ./{SAVE_DIR}/")

if __name__ == "__main__":
    main()
