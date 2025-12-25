import numpy as np
import pickle
from typing import List, Dict, Any, Tuple, Optional

# ---------------------------
# I/O helpers
# ---------------------------

def load_pkl(filepath: str):
    """Load a Python object from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Core utilities (keep structure & naming)
# ---------------------------

def random_sampling(n_subjects: int, sample_size: int, repeat_no: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Create a 2D matrix of indices:
    - shape: (repeat_no, sample_size)
    - each row is a unique random sample of subject indices (replace=False)
    """
    if sample_size > n_subjects:
        raise ValueError(f"sample_size={sample_size} exceeds available n_subjects={n_subjects}.")
    rng = np.random.default_rng(seed)
    idx_matrix = np.empty((repeat_no, sample_size), dtype=int)
    for r in range(repeat_no):
        idx_matrix[r] = rng.choice(n_subjects, size=sample_size, replace=False)
    return idx_matrix

def rmse(original: np.ndarray, new: np.ndarray) -> float:
    """Root-mean-square error between two equal-length arrays."""
    if original.shape != new.shape:
        raise ValueError(f"rmse shape mismatch: original {original.shape} vs new {new.shape}")
    diff = original - new
    return float(np.sqrt(np.mean(diff * diff)))

def _stack_axis_data(subject_axis_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Stack per-subject axis arrays into a 2D matrix of shape (n_subjects, 1001).
    Assumes each element is a 1-D array of length 1001.
    """
    return np.stack(subject_axis_arrays, axis=0)  # (S, 1001)

def compute_average_and_sd_across_samples_then_repeats(
    indices_matrix: np.ndarray,
    data2d: np.ndarray,
    ddof: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given:
      - indices_matrix: (repeats, sample_size)
      - data2d:         (n_subjects, 1001)
    Returns:
      - mean_across_attempts: (1001,) -> mean across samples then averaged across repeats
      - sd_across_attempts:   (1001,) -> sd across samples then averaged across repeats
    Steps (per repeat):
      1) map to (sample_size, 1001)
      2) mean over axis=0 -> (1001,)
      3) std  over axis=0 -> (1001,)
    Finally average these over repeats (axis=0).
    """
    repeats, sample_size = indices_matrix.shape
    # Map indices -> 3D array (repeats, sample_size, 1001)
    mapped = data2d[indices_matrix]  # advanced indexing; shape (R, N, 1001)

    # Mean and SD across the sample dimension (axis=1), per repeat -> (R, 1001)
    mean_per_repeat = mapped.mean(axis=1)                # (R, 1001)
    sd_per_repeat   = mapped.std(axis=1, ddof=ddof)      # (R, 1001)

    # Average across repeats (axis=0) -> (1001,)
    mean_across_attempts = mean_per_repeat.mean(axis=0)  # (1001,)
    sd_across_attempts   = sd_per_repeat.mean(axis=0)    # (1001,)
    return mean_across_attempts, sd_across_attempts

# ---------------------------
# Main routine
# ---------------------------

def run_sampling_stability(
    all_sample_mean_sd_per_feature: Dict[str, Dict[str, np.ndarray]],
    mean_per_subject: List[Dict[str, Dict[str, np.ndarray]]],
    featurelist: List[str],
    iteration_arr: np.ndarray,
    repeats_per_iteration: int = 20,
    rmse_threshold_mean: float = 0.05,   # tune as needed
    rmse_threshold_sd: float = 0.05,     # tune as needed
    ddof: int = 0,                       # 0: population SD, 1: sample SD
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    For each feature:
      - Randomly sample subjects across iterations (sample sizes).
      - Compute mean and SD of samples across repeats (X/Y/Z separately).
      - Compute RMSE vs ground-truth mean/sd.
      - Determine min_n per axis where both RMSE(mean) and RMSE(sd) <= thresholds.
      - Store RMSE per iteration and summary in pickletarget-like dict.
    """
    pickletarget: Dict[str, Any] = {}

    # Pre-extract per-feature per-axis per-subject arrays (to avoid re-allocations)
    # mean_per_subject is list of subjects; each subject[feature] has 'x','y','z'
    n_subjects = len(mean_per_subject)

    for feature in featurelist:
        # Collect per-subject arrays for this feature
        # all_subjects_data: [{x: np(1001), y: np(1001), z: np(1001)}, ...]
        all_subjects_data = [
            subj[feature] for subj in mean_per_subject
            if feature in subj
        ]
        if len(all_subjects_data) != n_subjects:
            raise ValueError(f"Feature '{feature}' missing for some subjects: "
                             f"found {len(all_subjects_data)} / expected {n_subjects}")

        # Stack X/Y/Z separately -> (S, 1001)
        data2d_x = _stack_axis_data([d['x'] for d in all_subjects_data])
        data2d_y = _stack_axis_data([d['y'] for d in all_subjects_data])
        data2d_z = _stack_axis_data([d['z'] for d in all_subjects_data])

        # Ground-truth arrays
        gt = all_sample_mean_sd_per_feature[feature]
        x_mean_gt, y_mean_gt, z_mean_gt = gt["x_mean"], gt["y_mean"], gt["z_mean"]
        x_sd_gt,   y_sd_gt,   z_sd_gt   = gt["x_sd"],   gt["y_sd"],   gt["z_sd"]

        # Per-iteration RMSEs stored for inspection
        rmse_per_iteration = {
            "mean": {"x": [], "y": [], "z": []},
            "sd":   {"x": [], "y": [], "z": []}
        }

        # Track min_n per axis once both mean & sd are stable
        min_n_x = None
        min_n_y = None
        min_n_z = None

        # Iterate sample sizes
        for n in iteration_arr:
            # Create indices for repeats
            indices_array = random_sampling(n_subjects, int(n), repeats_per_iteration, seed=seed)

            # Compute across samples -> repeats -> attempts (X/Y/Z)
            mean_x, sd_x = compute_average_and_sd_across_samples_then_repeats(indices_array, data2d_x, ddof=ddof)
            mean_y, sd_y = compute_average_and_sd_across_samples_then_repeats(indices_array, data2d_y, ddof=ddof)
            mean_z, sd_z = compute_average_and_sd_across_samples_then_repeats(indices_array, data2d_z, ddof=ddof)

            # RMSE vs ground-truth
            rmse_mean_x = rmse(x_mean_gt, mean_x)
            rmse_mean_y = rmse(y_mean_gt, mean_y)
            rmse_mean_z = rmse(z_mean_gt, mean_z)

            rmse_sd_x   = rmse(x_sd_gt, sd_x)
            rmse_sd_y   = rmse(y_sd_gt, sd_y)
            rmse_sd_z   = rmse(z_sd_gt, sd_z)

            # Store per iteration
            rmse_per_iteration["mean"]["x"].append((int(n), rmse_mean_x))
            rmse_per_iteration["mean"]["y"].append((int(n), rmse_mean_y))
            rmse_per_iteration["mean"]["z"].append((int(n), rmse_mean_z))

            rmse_per_iteration["sd"]["x"].append((int(n), rmse_sd_x))
            rmse_per_iteration["sd"]["y"].append((int(n), rmse_sd_y))
            rmse_per_iteration["sd"]["z"].append((int(n), rmse_sd_z))

            # Set min_n_* if not yet set and thresholds are met for BOTH mean & sd
            if min_n_x is None and (rmse_mean_x <= rmse_threshold_mean) and (rmse_sd_x <= rmse_threshold_sd):
                min_n_x = int(n)
            if min_n_y is None and (rmse_mean_y <= rmse_threshold_mean) and (rmse_sd_y <= rmse_threshold_sd):
                min_n_y = int(n)
            if min_n_z is None and (rmse_mean_z <= rmse_threshold_mean) and (rmse_sd_z <= rmse_threshold_sd):
                min_n_z = int(n)

        # Finalize summary for this feature
        # Choose max across axes so both mean & sd stability are respected
        min_candidates = [c for c in [min_n_x, min_n_y, min_n_z] if c is not None]
        min_n_total = max(min_candidates) if min_candidates else None

        pickletarget[feature] = {
            "min_n_x": min_n_x,
            "min_n_y": min_n_y,
            "min_n_z": min_n_z,
            "min_n_total": min_n_total,
            "rmse_mean_per_iteration": rmse_per_iteration["mean"],  # dict with lists of (n, rmse)
            "rmse_sd_per_iteration":   rmse_per_iteration["sd"],    # dict with lists of (n, rmse)
        }

    return pickletarget

# ---------------------------
# Example usage (adjust to your paths and data)
# ---------------------------

if __name__ == "__main__":
    # Example placeholders—replace with your actual file paths & feature list
    all_sample_mean_sd_per_feature = load_pkl("mean_data_abled_nature/all_mean_sd_perfeature.pkl.pkl")
    # mean_per_subject = load_pkl("path/to/mean_per_subject.pkl")
    # featurelist = ["HipAngles", "KneeAngles", ...]
    # iteration_arr = np.arange(1, 51)  # sample sizes from 1 to 50

    # result = run_sampling_stability(
    #     all_sample_mean_sd_per_feature=all_sample_mean_sd_per_feature,
    #     mean_per_subject=mean_per_subject,
    #     featurelist=featurelist,
    #     iteration_arr=iteration_arr,
    #     repeats_per_iteration=20,
    #     rmse_threshold_mean=0.05,
    #     rmse_threshold_sd=0.05,
    #     ddof=0,
    #     seed=42
    # )
    # print(result)
    pass