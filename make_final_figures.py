import os
import matplotlib.pyplot as plt

out_dir = "/home/chi-chi/REU/Final_Project/evaluation/graphs"
paper_dir = "/home/chi-chi/REU/Final_Project/paper/figures"
poster_dir = "/home/chi-chi/REU/Final_Project/poster/figures"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(paper_dir, exist_ok=True)
os.makedirs(poster_dir, exist_ok=True)

def savefig(name):
    for folder in [out_dir, paper_dir, poster_dir]:
        plt.savefig(os.path.join(folder, name), dpi=300, bbox_inches="tight")
    plt.close()

# 1. Clean performance before vs after defense
metrics = ["Precision", "Recall", "mAP50", "mAP50-95"]
baseline = [0.924, 0.952, 0.964, 0.695]
defense = [0.924, 0.947, 0.973, 0.703]

x = range(len(metrics))
plt.figure(figsize=(8, 5))
plt.bar([i - 0.2 for i in x], baseline, width=0.4, label="Baseline")
plt.bar([i + 0.2 for i in x], defense, width=0.4, label="Defense")
plt.xticks(x, metrics)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Clean Detection Performance Before vs After Defense")
plt.legend()
plt.grid(axis="y", alpha=0.3)
savefig("clean_performance_baseline_vs_defense.png")

# 2. Attack outcomes before vs after defense
attack_metrics = ["Successful Attacks", "Vanishing", "Unaffected"]
baseline_attack = [9, 159, 304]
defense_attack = [12, 121, 339]

x = range(len(attack_metrics))
plt.figure(figsize=(8, 5))
plt.bar([i - 0.2 for i in x], baseline_attack, width=0.4, label="Baseline")
plt.bar([i + 0.2 for i in x], defense_attack, width=0.4, label="Defense")
plt.xticks(x, attack_metrics)
plt.ylabel("Number of Images")
plt.title("Attack Outcomes Before vs After Defense")
plt.legend()
plt.grid(axis="y", alpha=0.3)
savefig("attack_outcomes_baseline_vs_defense.png")

# 3. Attack percentages
attack_percent_metrics = ["Attack Success", "Vanishing", "Total Affected"]
baseline_percent = [9/472*100, 159/472*100, (9+159)/472*100]
defense_percent = [12/472*100, 121/472*100, (12+121)/472*100]

x = range(len(attack_percent_metrics))
plt.figure(figsize=(8, 5))
plt.bar([i - 0.2 for i in x], baseline_percent, width=0.4, label="Baseline")
plt.bar([i + 0.2 for i in x], defense_percent, width=0.4, label="Defense")
plt.xticks(x, attack_percent_metrics)
plt.ylabel("Percent (%)")
plt.title("Adversarial Patch Impact Before vs After Defense")
plt.legend()
plt.grid(axis="y", alpha=0.3)
savefig("attack_percentages_baseline_vs_defense.png")

# 4. Per-class mAP50 comparison
classes = ["Red", "Yellow", "Green"]
baseline_map50 = [0.988, 0.946, 0.957]
defense_map50 = [0.989, 0.974, 0.955]

x = range(len(classes))
plt.figure(figsize=(8, 5))
plt.bar([i - 0.2 for i in x], baseline_map50, width=0.4, label="Baseline")
plt.bar([i + 0.2 for i in x], defense_map50, width=0.4, label="Defense")
plt.xticks(x, classes)
plt.ylim(0, 1.05)
plt.ylabel("mAP50")
plt.title("Per-Class Detection Performance")
plt.legend()
plt.grid(axis="y", alpha=0.3)
savefig("per_class_map50_baseline_vs_defense.png")

# 5. Overall affected reduction
labels = ["Baseline", "Defense"]
affected = [(9+159)/472*100, (12+121)/472*100]

plt.figure(figsize=(6, 5))
plt.bar(labels, affected)
plt.ylabel("Total Affected Images (%)")
plt.title("Overall Attack Impact Reduced After Defense")
plt.ylim(0, 45)
plt.grid(axis="y", alpha=0.3)
savefig("overall_attack_impact_reduction.png")

print("Final figures saved to:")
print(out_dir)
print(paper_dir)
print(poster_dir)
