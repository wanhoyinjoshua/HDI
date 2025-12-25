import matplotlib.pyplot as plt
import pickle
import numpy as np

def load_pkl(filepath: str):
    """Load a Python object from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Load your data
finalobject = load_pkl("mean_data_abled_nature/rmse_results.pkl")

for feature, data in finalobject.items():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

    # Collect all values to compute a shared y-axis limit
    all_vals = []

    for axis in ["x", "y", "z"]:
        samples = [t[0] for t in data["rmse_mean_per_iteration"][axis]]
        rmse_vals = np.array([t[1] for t in data["rmse_mean_per_iteration"][axis]])
        all_vals.append(rmse_vals)

        # Left subplot: Mean
        axs[0].plot(samples, rmse_vals, marker='o', label=f"{axis.upper()}")

        # Right subplot: SD
        if "rmse_sd_per_iteration" in data and axis in data["rmse_sd_per_iteration"]:
            rmse_sd = np.array([t[1] for t in data["rmse_sd_per_iteration"][axis]])
            all_vals.append(rmse_sd)
            axs[1].plot(samples, rmse_sd, marker='o', label=f"{axis.upper()} SD")

    # Standardize y-axis
    all_vals_flat = np.concatenate(all_vals)
    y_min, y_max = np.nanmin(all_vals_flat), np.nanmax(all_vals_flat)
    axs[0].set_ylim(y_min, y_max)
    axs[1].set_ylim(y_min, y_max)

    # Draw horizontal red lines at y=1,5,10
    for yline in [1, 5, 10]:
        axs[0].axhline(y=yline, color='red', linestyle='--', linewidth=1)
        axs[1].axhline(y=yline, color='red', linestyle='--', linewidth=1)

    # Titles and labels
    axs[0].set_title(f"{feature} Mean RMSE")
    axs[0].set_xlabel("Sample size")
    axs[0].set_ylabel("RMSE")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_title(f"{feature} RMSE SD")
    axs[1].set_xlabel("Sample size")
    axs[1].set_ylabel("RMSE")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{feature}_rmse_sidebyside.png", dpi=300)
    plt.close()
