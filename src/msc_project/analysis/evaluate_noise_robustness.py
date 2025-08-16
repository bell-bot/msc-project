import pandas as pd
import matplotlib.pyplot as plt
import os

def get_results_df(root_dir: str):

    dfs = []

    for _, dirs, _ in os.walk(root_dir):
        for dir in dirs:
            results_filename = os.path.join(root_dir, dir, "results.csv")
            if os.path.exists(results_filename):
                df = pd.read_csv(results_filename)
                dfs.append(df)

    return pd.concat(dfs)


    

results_df = get_results_df("results/noise_robustness")
results_df_sorted = results_df.sort_values("std", ignore_index=True)
results_df_truncated = results_df_sorted.iloc[5:]
fig, ax = plt.subplots(1,1)
ax.set_xscale("log")
ax.set_xlabel("Standard deviation")
ax.set_ylabel("Backdoor preservation rate")
ax.plot(results_df_truncated["std"], results_df_truncated[" preserve rate"])
fig.savefig("test.png")