import pandas as pd
import matplotlib.pyplot as plt

# Load your saved results
df = pd.read_csv("results_scored.csv")

# Create a bar chart of Average Quality Scores
summary = df.groupby("model")["quality_score"].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(summary["model"], summary["quality_score"], color=['blue', 'green', 'red'])
plt.title("Average Quality Score by Model")
plt.xlabel("Model")
plt.ylabel("Average Score (out of 5)")
plt.ylim(0, 5)

# Save the chart as a high-quality image for your dissertation
plt.savefig("quality_chart.png", dpi=300, bbox_inches='tight')
print("Chart saved as quality_chart.png!")