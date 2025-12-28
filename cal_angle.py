import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
file_path = 'able_body_cleaned_extracted_3d_angles.pkl' # Replace with your actual file path
#this is L initial contact first 
import numpy as np

import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def calculate_ankle_dorsiflexion_2d(data):
    """
    Calculate 2D ankle dorsiflexion/plantarflexion angles
    in the sagittal plane.

    Positive = dorsiflexion
    Negative = plantarflexion

    Parameters
    ----------
    data : dict
        Keys are joint names, each containing dicts with 'x' and 'z' arrays
        of shape (num_strides, num_frames)

    Returns
    -------
    ankle_angles : np.ndarray
        Shape (num_strides, num_frames)
    """

    knee_idx  = "R_KNE"
    ankle_idx = "R_ANK"
    heel_idx  = "R_HEE"
    toe_idx   = "R_TOE"

    num_strides = data[knee_idx]['x'].shape[0]
    num_frames  = data[knee_idx]['x'].shape[1]

    ankle_angles = np.zeros((num_strides, num_frames))

    for s in range(num_strides):

        # --- Extract sagittal-plane coordinates (X–Z) ---
        knee  = np.stack([data[knee_idx]['x'][s],  data[knee_idx]['z'][s]], axis=1)
        ankle = np.stack([data[ankle_idx]['x'][s], data[ankle_idx]['z'][s]], axis=1)
        heel  = np.stack([data[heel_idx]['x'][s],  data[heel_idx]['z'][s]], axis=1)
        toe   = np.stack([data[toe_idx]['x'][s],   data[toe_idx]['z'][s]], axis=1)

        # --- Segment vectors ---
        shank = ankle - knee        # knee → ankle
        foot  = toe - heel          # heel → toe (IMPORTANT)

        # Normalize
        shank_unit = shank / np.linalg.norm(shank, axis=1, keepdims=True)
        foot_unit  = foot  / np.linalg.norm(foot,  axis=1, keepdims=True)

        # --- Unsigned angle between segments ---
        dot = np.sum(shank_unit * foot_unit, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        theta = np.degrees(np.arccos(dot))  # ≈ 90° at neutral

        # --- Signed dorsiflexion ---
        # 2D cross product (sagittal plane sign)
        cross =  shank_unit[:, 1] * foot_unit[:, 0]- shank_unit[:, 0] * foot_unit[:, 1] 
        sign = np.sign(cross)

        # Clinical ankle angle
        ankle_angles[s, :] = sign * (90.0 - theta)

        # --- Plot ---
        plt.figure(figsize=(10, 4))
        plt.plot(ankle_angles[s], label="Ankle dorsiflexion")
        plt.axhline(0, linestyle="--", color="gray")
        plt.xlabel("Frame")
        plt.ylabel("Angle (deg)")
        plt.title(f"Stride {s}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"stride{s}_ankle_dorsiflexion_2d.png", dpi=300)
        plt.close()

    return ankle_angles



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




import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def calculate_hip_angles_sagittal_2d_3dpelvis(data):
    """
    Sagittal-plane hip flexion/extension angle (X-Z plane)
    using a vertical reference vector derived from 3D pelvis axes.
    
    Parameters
    ----------
    data : dict
        Keys are joint names, each containing dicts with 'x', 'y', 'z' arrays
        of shape (num_strides, num_frames)
    
    Returns
    -------
    hip_angles : np.ndarray
        Array of shape (num_strides, num_frames) with hip flexion/extension angles in degrees
    """

    pelvis_o = "B_PELO"  # Pelvis origin
    sacrum   = "B_SACR"
    R_ASI    = "R_ASI"
    L_ASI    = "L_ASI"
    hip_idx  = "R_HJC"
    knee_idx = "R_KNE"

    num_strides = data[hip_idx]['x'].shape[0]
    num_frames  = data[hip_idx]['x'].shape[1]

    hip_angles = np.zeros((num_strides, num_frames))

    for s in range(num_strides):
        # --- 3D pelvis markers ---
        RASI = np.stack([data[R_ASI]['x'][s], data[R_ASI]['y'][s], data[R_ASI]['z'][s]], axis=1)
        LASI = np.stack([data[L_ASI]['x'][s], data[L_ASI]['y'][s], data[L_ASI]['z'][s]], axis=1)
        SACR = np.stack([data[sacrum]['x'][s], data[sacrum]['y'][s], data[sacrum]['z'][s]], axis=1)
        PELO = np.stack([data[pelvis_o]['x'][s], data[pelvis_o]['y'][s], data[pelvis_o]['z'][s]], axis=1)

        # --- Construct 3D pelvis axes ---
        pelvis_ml_axis = RASI - LASI  # medio-lateral
        pelvis_mid_asi = 0.5 * (RASI + LASI)
        pelvis_ap_axis = pelvis_mid_asi - SACR  # antero-posterior
        pelvis_vert_axis = np.cross(pelvis_ml_axis, pelvis_ap_axis)  # proximal-distal

        # --- Take only sagittal component (X-Z) of vertical axis ---
        pelvis_vert_2d = np.stack([pelvis_vert_axis[:,0], pelvis_vert_axis[:,2]], axis=1)
        pelvis_unit = pelvis_vert_2d / np.linalg.norm(pelvis_vert_2d, axis=1, keepdims=True)

        # --- Thigh vector in sagittal plane (X-Z) ---
        hip  = np.stack([data[hip_idx]['x'][s], data[hip_idx]['z'][s]], axis=1)
        knee = np.stack([data[knee_idx]['x'][s], data[knee_idx]['z'][s]], axis=1)
        thigh_vec = knee - hip
        thigh_unit = thigh_vec / np.linalg.norm(thigh_vec, axis=1, keepdims=True)

        # --- Compute signed 2D angle ---
        cross = pelvis_unit[:,0]*thigh_unit[:,1] - pelvis_unit[:,1]*thigh_unit[:,0]
        dot   = np.sum(pelvis_unit * thigh_unit, axis=1)
        theta = np.arctan2(cross, dot)

        hip_angles[s,:] = np.degrees(theta)

        # --- Plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, num_frames + 1), hip_angles[s])
        plt.axhline(0, linestyle="--", color='gray')
        plt.xlabel("Frame")
        plt.ylabel("Hip flexion/extension (deg)")
        plt.title(f"Stride {s}")
        plt.savefig(f"stride{s}_hip_angles_sagittal_2d.png", dpi=300, bbox_inches="tight")
        plt.close()

    return hip_angles



import numpy as np
import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt

def calculate_hip_angles_frontal_simple(data):
    """
    Frontal-plane hip abduction/adduction angle (Y-Z plane)
    using the pelvis ML axis and thigh vector directly.
    
    Positive = adduction, negative = abduction
    
    Parameters
    ----------
    data : dict
        Keys are joint names, each containing dicts with 'x', 'y', 'z' arrays
        of shape (num_strides, num_frames)
    
    Returns
    -------
    hip_angles : np.ndarray
        Array of shape (num_strides, num_frames) with hip abduction/adduction angles in degrees
    """

    R_ASI = "R_ASI"
    L_ASI = "L_ASI"
    hip_idx = "R_HJC"
    knee_idx = "R_KNE"

    num_strides = data[hip_idx]['x'].shape[0]
    num_frames = data[hip_idx]['x'].shape[1]

    hip_angles = np.zeros((num_strides, num_frames))

    for s in range(num_strides):
        # --- Pelvis ML axis in frontal plane ---
        RASI = np.stack([data[R_ASI]['y'][s], data[R_ASI]['z'][s]], axis=1)
        LASI = np.stack([data[L_ASI]['y'][s], data[L_ASI]['z'][s]], axis=1)
        pelvis_ml = RASI - LASI
        pelvis_unit = pelvis_ml / np.linalg.norm(pelvis_ml, axis=1, keepdims=True)

        # --- Thigh vector in frontal plane ---
        hip  = np.stack([data[hip_idx]['y'][s], data[hip_idx]['z'][s]], axis=1)
        knee = np.stack([data[knee_idx]['y'][s], data[knee_idx]['z'][s]], axis=1)
        thigh_vec = knee - hip
        thigh_unit = thigh_vec / np.linalg.norm(thigh_vec, axis=1, keepdims=True)

        # --- Compute signed angle (2D cross/dot) ---
        cross = pelvis_unit[:,0]*thigh_unit[:,1] - pelvis_unit[:,1]*thigh_unit[:,0]
        dot   = np.sum(pelvis_unit * thigh_unit, axis=1)
        theta = np.arctan2(cross, dot)

        hip_angles[s,:] = np.degrees(theta)

        # --- Plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, num_frames + 1), hip_angles[s])
        plt.axhline(0, linestyle="--", color='gray')
        plt.xlabel("Frame")
        plt.ylabel("Hip abduction/adduction (deg)")
        plt.title(f"Stride {s}")
        plt.savefig(f"stride{s}_hip_angles_frontal_simple.png", dpi=300, bbox_inches="tight")
        plt.close()

    return hip_angles



def calculate_hip_angles_frontal_3d(data):
    """
    Full 3D hip abduction/adduction angle using pelvis vertical axis.

    Parameters
    ----------
    data : dict
        Keys are joint names, each containing dicts with 'x', 'y', 'z' arrays
        of shape (num_strides, num_frames)

    Returns
    -------
    hip_angles : np.ndarray
        Array of shape (num_strides, num_frames) with hip abduction/adduction angles in degrees
    """

    pelvis_o = "B_PELO"  # Pelvis origin
    sacrum   = "B_SACR"
    R_ASI    = "R_ASI"
    L_ASI    = "L_ASI"
    hip_idx  = "R_HJC"
    knee_idx = "R_KNE"

    num_strides = data[hip_idx]['x'].shape[0]
    num_frames  = data[hip_idx]['x'].shape[1]

    hip_angles = np.zeros((num_strides, num_frames))

    for s in range(num_strides):
        # --- 3D pelvis markers ---
        RASI = np.stack([data[R_ASI]['x'][s], data[R_ASI]['y'][s], data[R_ASI]['z'][s]], axis=1)
        LASI = np.stack([data[L_ASI]['x'][s], data[L_ASI]['y'][s], data[L_ASI]['z'][s]], axis=1)
        SACR = np.stack([data[sacrum]['x'][s], data[sacrum]['y'][s], data[sacrum]['z'][s]], axis=1)
        PELO = np.stack([data[pelvis_o]['x'][s], data[pelvis_o]['y'][s], data[pelvis_o]['z'][s]], axis=1)

        # --- Construct 3D pelvis axes ---
        pelvis_ml_axis = RASI - LASI
        pelvis_mid_asi = 0.5 * (RASI + LASI)
        pelvis_ap_axis = pelvis_mid_asi - SACR
        pelvis_vert_axis = np.cross(pelvis_ml_axis, pelvis_ap_axis)  # vertical (proximal-distal)
        pelvis_unit = pelvis_vert_axis / np.linalg.norm(pelvis_vert_axis, axis=1, keepdims=True)

        # --- Thigh vector ---
        hip  = np.stack([data[hip_idx]['x'][s], data[hip_idx]['y'][s], data[hip_idx]['z'][s]], axis=1)
        knee = np.stack([data[knee_idx]['x'][s], data[knee_idx]['y'][s], data[knee_idx]['z'][s]], axis=1)
        thigh_vec = knee - hip
        thigh_unit = thigh_vec / np.linalg.norm(thigh_vec, axis=1, keepdims=True)

        # --- Compute angle between pelvis vertical axis and thigh vector ---
        dot = np.sum(pelvis_unit * thigh_unit, axis=1)
        dot = np.clip(dot, -1.0, 1.0)  # avoid numerical errors
        theta = np.arccos(dot)  # radians

        hip_angles[s, :] = np.degrees(theta)

        # --- Plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, num_frames + 1), hip_angles[s])
        plt.axhline(0, linestyle="--", color='gray')
        plt.xlabel("Frame")
        plt.ylabel("Hip abduction/adduction (deg)")
        plt.title(f"Stride {s}")
        plt.savefig(f"stride{s}_hip_angles_frontal_3d.png", dpi=300, bbox_inches="tight")
        plt.close()

    return hip_angles


# knee_angles = calculate_knee_angle(data)
# print(knee_angles.shape)  # should be (1001,)
with open(file_path, "rb") as f:
    subjects = pickle.load(f)
    #data = pickle.load(subject)
    #data= calculate_knee_angles_2d(subjects[0])
    #hip = calculate_hip_angles_sagittal_2d_3dpelvis(subjects[0])
    #hip_ab = calculate_hip_angles_frontal_simple(subjects[0])
    #er=calculate_hip_angles_frontal_3d(subjects[0])
    ankle =calculate_ankle_dorsiflexion_2d(subjects[0])
    