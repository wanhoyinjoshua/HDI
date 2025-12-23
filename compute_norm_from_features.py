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
    #for each subjec avg of all strides 
    error_data={}
   
    for feature in features:
        print(data[feature])
        
        result= compute_xyz_mean_sd(data[feature])
        all_mean_arr_x = np.empty((0, 1001))
        all_mean_arr_y = np.empty((0, 1001))
        all_mean_arr_z = np.empty((0, 1001))
        all_mean_arr_x_sd = np.empty((0, 1001))
        all_mean_arr_y_sd = np.empty((0, 1001))
        all_mean_arr_z_sd = np.empty((0, 1001))
        for subject in subjects:
         
            avg_sub_mean_x= np.mean(subject[feature]["x"],axis=0)
            
            avg_sub_mean_x_row = avg_sub_mean_x.reshape(1, -1)
            all_mean_arr_x =np.vstack([all_mean_arr_x, avg_sub_mean_x_row])
           
            avg_sub_mean_y= np.mean(subject[feature]["y"],axis=0)
            avg_sub_mean_y_row = avg_sub_mean_y.reshape(1, -1)
            all_mean_arr_y =np.vstack([all_mean_arr_y, avg_sub_mean_y_row])

            avg_sub_mean_z= np.mean(subject[feature]["z"],axis=0)
            avg_sub_mean_z_row = avg_sub_mean_z.reshape(1, -1)
            all_mean_arr_z=np.vstack([all_mean_arr_z, avg_sub_mean_z_row])
        

       
        #plot(feature,result)
        #need to select sample 
        experiment= list(range(1, 139))
        repeat_exp=20
        exp_result={}
        for exp in experiment:
            print("fuck yea new exp")
           
            num_rows = 20
            exp = 1001  # number of elements per row
            all_samples = np.empty((num_rows, exp), dtype=int)
            for i in range(num_rows):
                random_samples_indices = random.choices(experiment, k=exp)
                print(i)
                for sam in random_samples_indices:
                    print(sam)
                    sam = [sam - 1] if isinstance(sam, int) else sam-1
                    print(rmse(np.mean(all_mean_arr_x,axis=0),np.mean(all_mean_arr_x[sam, :],axis=0)))
                    #now same with sd 
                    #compare 
                
                  
            # now i have the 20 rows of data selected 

            #calculate mean and sd with random sample 
            #repeat exp 20 times 
            # get the avergae of that and sd
            # compare it with the gold standard 
            # get the error score 
            #populate the error data

        """
        i need a function to intake seed no and the sbjects data 
        and output a computed avg and sd for bth x y and z for given feature 
        so it will just be time series data
        for subject in subjects:
            combined_x_norm
            combined_x_sd
            combined_y_norm
            combined_y_sd
            combined_z_norm
            combined_zsd 
        """
        
        
        