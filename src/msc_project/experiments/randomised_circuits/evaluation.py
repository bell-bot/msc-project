import pandas as pd

baseline_results = "results/random_circuit/baseline_normal_small/evaluation_report.csv"
experiment_results = "results/random_circuit/experiment_normal_small/evaluation_report.csv"

baseline_df = pd.read_csv(baseline_results)
experiment_df = pd.read_csv(experiment_results)

print(baseline_df.mean(numeric_only=True))
print("---")
print(experiment_df.mean(numeric_only=True))