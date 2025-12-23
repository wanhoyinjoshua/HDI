import numpy as np

def fill_missing_values(axis_dict):
    """
    Replace NaNs in each axis array with column-wise mean.
    axis_dict: {'x': ndarray, 'y': ndarray, 'z': ndarray} or {'n': ndarray}
    Returns: cleaned axis_dict with NaNs replaced.
    """
    cleaned = {}
    for axis, arr in axis_dict.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]  # upcast to 2D
        if np.isnan(arr).any():
            col_mean = np.nanmean(arr, axis=0)  # mean per time index
            inds = np.where(np.isnan(arr))
            arr[inds] = col_mean[inds[1]]       # fill NaNs with column mean
        cleaned[axis] = arr
    return cleaned

def extract_features(list_of_feature_names, data):
    """
    Flatten nested L_segment structure into a simple dict per subject.
    Input:
        list_of_feature_names: e.g. ["hip_angles", "knee_angles", "bilateral_pelvis"]
        data: list of subject dicts with structure:
              subject["L_segment"]["L_segment_L"][feature] -> {x:..., y:..., z:...}
              subject["L_segment"]["L_segment_R"][feature] -> {x:..., y:..., z:...}
              subject["L_segment"]["L_segment_B"][feature] -> {x:..., y:..., z:...}
    Output:
        list of dicts with keys like:
            "L_<feature>", "R_<feature>", "B_<feature>"
            values are axis dicts {x: ndarray, y: ndarray, z: ndarray}
    """
    output = []


    subject_data = {}
    #print("processing start")

    for feature in list_of_feature_names:
        # Handle bilateral
        if "bilateral" in feature:
            # Extract name after 'bilateral_'
            parts = feature.split("_", 1)
            base_name = parts[1] if len(parts) > 1 else feature
            B_container = data["LsideSegm"]["Bside"]
            subject_data[f"B_{base_name}"] = B_container.get(base_name)
            continue

        # Left side
        L_container = data["LsideSegm"]["Lside"]
        subject_data[f"L_{feature}"] = L_container.get(feature)

        # Right side
        R_container = data["LsideSegm"]["Rside"]
        subject_data[f"R_{feature}"] = R_container.get(feature)

    return(subject_data)
    #print("processing end")

 


features = [
    "HipAngles",
    "KneeAngles",
    "AnkleAngles",
    "PelvisAngles",
    
    "ShoulderAngles",
    "ElbowAngles",
    "WristAngles",
    "NeckAngles",
    "SpineAngles",
    "HeadAngles",
    "bilateral_CentreOfMass",
    "ThoraxAngles"
]
import pickle
file_path = 'able_body_pkl_nature2.pkl' # Replace with your actual file path
print("suckers")
import os
cwd = os.getcwd()
print(cwd)
PKL_DIR = os.path.join(cwd,"data")   # <-- change this
PKL_EXT = ".pkl"

import os
final_result=[]
for fname in sorted(os.listdir(PKL_DIR)):
    if not fname.endswith(PKL_EXT):
        continue
    

    file_path = os.path.join(PKL_DIR, fname)
    print(f"Loading {file_path}")

    with open(file_path, "rb") as f:
        subject = pickle.load(f)
        #data = pickle.load(subject)
        print(subject["LsideSegm"]["Bside"].keys())
        print(subject["LsideSegm"]["Lside"].keys())
        print(subject["LsideSegm"]["Rside"].keys())
    
        result= extract_features(features,subject)
        print(result)
        #now deal with nan
       
        for key, axis_dict in result.items():
            if axis_dict is not None:
                result[key] = fill_missing_values(axis_dict)

        final_result.append(result)

output_path = "able_body_cleaned_extracted_3d_angles.pkl"

with open(output_path, "wb") as f:
    pickle.dump(final_result, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {output_path}")




