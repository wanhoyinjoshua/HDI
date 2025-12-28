import numpy as np

def fill_missing_values(axis_dict):
    """
    Replace NaNs in each axis array with column-wise mean.
    axis_dict: {'x': ndarray, 'y': ndarray, 'z': ndarray} or {'n': ndarray}
    Returns: cleaned axis_dict with NaNs replaced.
    """
    print(axis_dict)
    if isinstance(axis_dict, (dict, np.ndarray)) and isinstance(axis_dict, dict):
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
    else:
        return axis_dict

import numpy as np

def cal_HJC(sub_data, mm=0):
    """
    Calculate Left and Right Hip Joint Centers (HJC) using Newington-Gage method
    with fixed theta and beta angles.

    Parameters
    ----------
    sub_data : dict
        Expected keys:
            - 'L_ASI', 'R_ASI': np.ndarray of shape (num_frames,)
            - 'L_ASI_y', 'R_ASI_y': np.ndarray for Y (lateral) positions
            - 'L_ASI_z', 'R_ASI_z': np.ndarray for Z (vertical) positions
            - 'char_LegLength': scalar
    mm : float
        Marker radius offset (optional, default=0)

    Returns
    -------
    HJC_L : np.ndarray of shape (num_frames, 3)
    HJC_R : np.ndarray of shape (num_frames, 3)
    """

    # Extract markers
    L_ASI_x = np.array(sub_data["L_ASI"]["x"])
    R_ASI_x = np.array(sub_data["R_ASI"]["x"])
    L_ASI_y = np.array(sub_data["L_ASI"]["y"])
    R_ASI_y = np.array(sub_data["R_ASI"]["y"])
    L_ASI_z = np.array(sub_data["L_ASI"]["z"])
    R_ASI_z = np.array(sub_data["R_ASI"]["z"])

    # 1. Inter-ASIS distance and midpoint
    interasis = np.sqrt((L_ASI_x - R_ASI_x)**2 +
                        (L_ASI_y - R_ASI_y)**2 +
                        (L_ASI_z - R_ASI_z)**2)
    aa = interasis / 2

    # 2. ASIS to trochanter distance
    LegLength = sub_data["char_LegLength"]
    AsisTrocDist = 0.1288 * LegLength - 48.56

    # 3. Constant C
    C = LegLength * 0.115 - 15.3

    # 4. Fixed angles
    theta = 0.5   # radians
    beta = 0.314  # radians

    # 5. Offsets
    X_offset = C * np.cos(theta) * np.sin(beta) - (AsisTrocDist + mm) * np.cos(beta)
    Y_offset = C * np.sin(theta) - aa
    Z_offset = -C * np.cos(theta) * np.cos(beta) - (AsisTrocDist + mm) * np.sin(beta)

    # 6. Pelvis origin
    origin_X = (L_ASI_x + R_ASI_x) / 2
    origin_Y = (L_ASI_y + R_ASI_y) / 2
    origin_Z = (L_ASI_z + R_ASI_z) / 2

    # 7. Left and Right HJC
    HJC_L = np.stack([origin_X + X_offset,
                      origin_Y + Y_offset,
                      origin_Z + Z_offset], axis=1)

    HJC_R = np.stack([origin_X + X_offset,
                      origin_Y - Y_offset,  # Y offset negated for right hip
                      origin_Z + Z_offset], axis=1)

    return HJC_L, HJC_R


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
    print(data.keys())
    for feature in list_of_feature_names:
        # Handle bilateral
        if "char" in feature:
            parts = feature.split("_", 1)
            base_name = parts[1] if len(parts) > 1 else feature
            B_container = data.get("sub_char", {})  # ensure sub_char exists
            # Always create the field; if it doesn't exist, value will be None
            subject_data[f"char_{base_name}"] = B_container.get(base_name, None)
            continue

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
    "ASI",
    "KNE",
    "bilateral_SACR",
    "bilateral_PELP",
    "bilateral_PELO",

    
    #"ShoulderAngles",
    #"ElbowAngles",
    #"WristAngles",
    #"NeckAngles",
    "SpineAngles",
    #"HeadAngles",
    #"bilateral_CentreOfMass",
    "ThoraxAngles",
    "char_Weight",
    "char_Age",
    "char_Male",
    "char_Height",
    "char_LegLength",
    "char_TPS",
    "char_LesionLeft",
    "char_FAC",
    "char_POMA",
    "char_TIS"
    
]
import pickle
file_path = 'able_body_pkl_nature2.pkl' # Replace with your actual file path
#this file has alreadt computed the avg of different strides 
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
        print(subject["sub_char"].keys())
        print(subject["LsideSegm"]["Bside"].keys())
        #print(subject["LsideSegm"]["Lside"].keys())
        #print(subject["LsideSegm"]["Rside"].keys())
    
        result= extract_features(features,subject)
      
        #now deal with nan
       
        for key, axis_dict in result.items():
            if axis_dict is not None:
                result[key] = fill_missing_values(axis_dict)

        
        #ok now here need to calculate 
        #InterAsis distance
        # e mean distance between the LASI and RASI markers
        # Asis to Trocanter distances AsisTrocDist = 0.1288 * LegLength – 48.56
        #C = MeanLegLength*0.115 – 15.3
        #aa is half the InterAsis distance
        # X = C*cos(theta)*sin(beta) – (AsisTrocDist + mm) * cos(beta)
        #Y = -(C*sin(theta) – aa)
        #Z = -C*cos(theta)*cos(beta) – (AsisTrocDist + mm) * sin(beta)

        
        result["L_HJC"]=cal_HJC(result)[0]
        result["R_HJC"]=cal_HJC(result)[1]
        print("joint center")
        print(result["L_HJC"].shape)
        print(result["L_HJC"])

        final_result.append(result)

output_path = "able_body_cleaned_extracted_3d_angles.pkl"

with open(output_path, "wb") as f:
    pickle.dump(final_result, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {output_path}")




