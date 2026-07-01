import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the results
csv_path = "perturb_output/covid_monocytes.csv"
df = pd.read_csv(csv_path)

# 2. Sort by the shift magnitude
# We want the genes that caused the largest POSITIVE shift toward the goal (Normal)
top_rescuers = df.sort_values(by="Shift_to_goal_end", ascending=False).head(10)

# 3. Create the Visualization for your Slides
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_rescuers, 
    x="Shift_to_goal_end", 
    y="Gene_name", 
    palette="Reds_r" # Dark red for highest targets
)

plt.title("Top 10 Therapeutic Targets for COVID-19 (CD14+ Monocytes)", fontsize=14)
plt.xlabel("Shift toward Normal State (Probability Change)", fontsize=12)
plt.ylabel("Gene Name", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig("top_targets_plot.png", dpi=300)
print("Success! Plot saved as top_targets_plot.png")

# 4. Print the table for your slide
print("\n--- DATA FOR YOUR SLIDES ---")
print(top_rescuers[["Gene_name", "Ensembl_ID", "Shift_to_goal_end", "N_Detections"]])