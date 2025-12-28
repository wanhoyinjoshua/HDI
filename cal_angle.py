import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
file_path = 'able_body_cleaned_extracted_3d_angles.pkl' # Replace with your actual file path
#this is L initial contact first 
import numpy as np

import numpy as np

def calculate_knee_angles_2d(data):
    """
    Calculate 2D knee angles for all strides.
    
    Parameters
    ----------
    data : dict
        Keys are joint names, each containing dicts with 'x' and 'y' arrays of shape (num_strides, num_frames)
        Example: data['L_HJC']['x'].shape -> (7, 1001)
    
    Returns
    -------
    knee_angles : np.ndarray
        Array of shape (num_strides, num_frames) with knee angles in degrees
    """
    hip_idx = "R_HJC"
    knee_idx = "R_KNE"
    ankle_idx = "R_ANK"

    num_strides = data[hip_idx]['x'].shape[0]
    num_frames = data[hip_idx]['x'].shape[1]

    knee_angles = np.zeros((num_strides, num_frames))

    for s in range(num_strides):
        # Stack x and y coordinates for each joint to shape (num_frames, 2)
        hip = np.stack([data[hip_idx]['x'][s], data[hip_idx]['z'][s]], axis=1)
        knee = np.stack([data[knee_idx]['x'][s], data[knee_idx]['z'][s]], axis=1)
        ankle = np.stack([data[ankle_idx]['x'][s], data[ankle_idx]['z'][s]], axis=1)

        # Vectors
        vec_hip_to_knee = knee - hip
        vec_knee_to_ankle = ankle - knee

        # Dot product and norms
        dot_prod = np.sum(vec_hip_to_knee * vec_knee_to_ankle, axis=1)
        norm1 = np.linalg.norm(vec_hip_to_knee, axis=1)
        norm2 = np.linalg.norm(vec_knee_to_ankle, axis=1)

        # Angle in radians
        cos_theta = dot_prod / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)

        # Convert to degrees
        knee_angles[s, :] = np.degrees(theta_rad)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, 1001 + 1),knee_angles[s, :], label=f'Stride ')
        plt.savefig(f"stride{s}knee_angles_plot_2d.png", dpi=300, bbox_inches='tight')

    return knee_angles



# knee_angles = calculate_knee_angle(data)
# print(knee_angles.shape)  # should be (1001,)
with open(file_path, "rb") as f:
    subjects = pickle.load(f)
    #data = pickle.load(subject)
    data= calculate_knee_angles_2d(subjects[0])
    print(data)
    mean=np.mean(data,0)
    num_frames = 1001
    x = np.arange(1, 1001 + 1)  # 1 to 1001

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, label=f'Stride ')
    plt.savefig('knee_angles_plot_2d.png', dpi=300, bbox_inches='tight')
