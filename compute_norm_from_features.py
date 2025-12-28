import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
file_path = 'able_body_cleaned_extracted_3d_angles.pkl' # Replace with your actual file path
#this is L initial contact first 

def rmse(ts1, ts2):
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    
    if ts1.shape != ts2.shape:
        raise ValueError("Time series must have the same shape")
    
    return np.sqrt(np.mean((ts1 - ts2) ** 2))

features = [
    "L_HipAngles",
    "R_HipAngles",
    "L_KneeAngles",
    "R_KneeAngles",
    "L_AnkleAngles",
    "R_AnkleAngles",
    "L_PelvisAngles",
    "R_PelvisAngles",
    "R_ASI",
    "R_KNE",
    "L_ASI",
    "L_KNE",
    #"B_SACR",
    "B_PELP",
    "B_PELO",
    
    "L_ShoulderAngles",
    "R_ShoulderAngles",
    "L_ElbowAngles",
    "R_ElbowAngles",
    "L_WristAngles",
    "R_WristAngles",
    "L_NeckAngles",
    "R_NeckAngles",
    "L_SpineAngles",
    "R_SpineAngles",
    "L_HeadAngles",
    "R_HeadAngles",
    "B_CentreOfMass",
    "L_ThoraxAngles",
    "R_ThoraxAngles"
]

def compute_xyz_mean_sd(data):
    newdata={}
    newdata["x"]=np.mean(data["x"], axis=0)
    newdata["x_sd"]=np.std(data["x"], axis=0)
    newdata["y"]=np.mean(data["y"], axis=0)
    newdata["y_sd"]=np.std(data["y"], axis=0)
    newdata["z"]=np.mean(data["z"], axis=0)
    newdata["z_sd"]=np.std(data["z"], axis=0)
    return newdata

def plot(title, data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["x", "y", "z"]
    
    for index, axis_label in enumerate(labels):
        time = np.arange(1, 1002)

        axes[index].plot(time, data[axis_label], color='blue', label='Mean')
        axes[index].fill_between(time, 
                                 data[axis_label] - data[f"{axis_label}_sd"], 
                                 data[axis_label] + data[f"{axis_label}_sd"],
                                 color='blue', alpha=0.2, label='±1 SD')
        axes[index].set_title(f'{title} Plot {axis_label} axis')
        axes[index].set_xlabel('Time Points (normalised)')
        axes[index].set_ylabel('Angle (deg)')
        axes[index].grid(True)
        axes[index].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{title}plot.png")




def compute_mean_sd(timeseries):
    """
    timeseries: np.ndarray
        Shape: (N, 1001) OR (subjects, strides, 1001)

    Returns:
        mean_ts: (1001,)
        sd_ts:   (1001,)
    """

    # If 3D: collapse subjects + strides
    if timeseries.ndim == 3:
        timeseries = timeseries.reshape(-1, timeseries.shape[-1])

    mean_ts = np.nanmean(timeseries, axis=0)
    sd_ts   = np.nanstd(timeseries, axis=0)

    return mean_ts, sd_ts


with open(file_path, "rb") as f:
    subjects = pickle.load(f)
    #data = pickle.load(subject)

    data=subjects[0]
    
    print(data["R_ASI"])
    print(data["R_KNE"])
    print(data["L_ASI"])
    print(data["L_KNE"])
    print(data["B_SACR"])
    print(data["B_PELP"])
    

    if len(subjects) > 115:
        subjects.pop(115)  # removes the 116th element
    #for each subjec avg of all strides 
    error_data={}
    final_mean=[]
    final_sd=[]
    finalobject={}
    subjectobject = {}

   
    for feature in features:
      
        individual_sub_data={}
        all_mean_arr_x = np.empty((0, 1001))
        all_mean_arr_y = np.empty((0, 1001))
        all_mean_arr_z = np.empty((0, 1001))
        for subject_id, subject in enumerate(subjects):
            #this is at a subject level
            sid = f"sub_{subject_id:03d}"
            subjectobject.setdefault(sid, {})
            subjectobject[sid].setdefault(feature, {})
            print(feature)
            print(subject_id)

            if subject[feature]["x"] is not None:
                arr = subject[feature]["x"]
                
            else:
                arr = np.array([])
            nan_count = np.isnan(arr).sum()
         
            
            if nan_count > 1:
                print(f"nan count in feature_{feature}: {nan_count}")
                print(subject_id)
            avg_sub_mean_x= np.mean(subject[feature]["x"],axis=0)
            avg_sub_mean_x_row = avg_sub_mean_x.reshape(1, -1)
            all_mean_arr_x =np.vstack([all_mean_arr_x, avg_sub_mean_x_row])  
            subjectobject[sid][feature]["mean_x"] = avg_sub_mean_x
            subjectobject[sid][feature]["sd_x"] = np.std(subject[feature]["x"],axis=0)
        
           
            avg_sub_mean_y= np.mean(subject[feature]["y"],axis=0)
            avg_sub_mean_y_row = avg_sub_mean_y.reshape(1, -1)
            all_mean_arr_y =np.vstack([all_mean_arr_y, avg_sub_mean_y_row])
            subjectobject[sid][feature]["mean_y"] = avg_sub_mean_y
            subjectobject[sid][feature]["sd_y"] = np.std(subject[feature]["y"],axis=0)
        


            avg_sub_mean_z= np.mean(subject[feature]["z"],axis=0)
            avg_sub_mean_z_row = avg_sub_mean_z.reshape(1, -1)
            all_mean_arr_z=np.vstack([all_mean_arr_z, avg_sub_mean_z_row])
            subjectobject[sid][feature]["mean_z"] = avg_sub_mean_z
            subjectobject[sid][feature]["sd_z"] = np.std(subject[feature]["z"],axis=0)
        



        finalobject.setdefault(feature, {})
        
        finalobject[feature]["mean_x"] = np.mean(all_mean_arr_x, axis=0)
        finalobject[feature]["sd_x"]   = np.std(all_mean_arr_x, axis=0)
        finalobject[feature]["mean_y"] = np.mean(all_mean_arr_y, axis=0)
        finalobject[feature]["sd_y"]   = np.std(all_mean_arr_y, axis=0)
        finalobject[feature]["mean_z"] = np.mean(all_mean_arr_z, axis=0)
        finalobject[feature]["sd_z"]   = np.std(all_mean_arr_z, axis=0)
        
       
       
    import pickle
    #so this is per subject.... it contains an array, each array contains 138 subjects...
    #so i need to save two files one being 
    # ah and so the sd here is at a subject level
    # what i need is the the total mean, and the sd for that 

    with open("mean_data_abled_nature/all_mean_sd_perfeature.pkl", "wb") as f:
        pickle.dump(finalobject, f, protocol=pickle.HIGHEST_PROTOCOL)
        #need to do sd still

    with open("mean_data_abled_nature/sub_level_mean_sd_allfeat.pkl", "wb") as f:
        pickle.dump(subjectobject, f, protocol=pickle.HIGHEST_PROTOCOL)
        #need to do sd still
        