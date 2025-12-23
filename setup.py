import h5py
import numpy as np

file_path = "/mnt/d/MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat"


def deref(f, obj):
    """
    Fully dereference MATLAB v7.3 HDF5 objects into Python objects.
    """
    if isinstance(obj, h5py.Dataset):
        if obj.dtype == object:
            # MATLAB object references
            out = []
            for i in range(obj.shape[0]):
                ref = obj[i, 0]
                out.append(deref(f, f[ref]))
            return out
        else:
            return np.array(obj)

    elif isinstance(obj, h5py.Group):
        return {k: deref(f, obj[k]) for k in obj.keys()}

    else:
        return obj


def load_all_subjects(file_path):
    subjects = []

    with h5py.File(file_path, "r") as f:
        sub = f["/Sub"]
        print(sub.keys())

        # number of subjects inferred from one dataset
        n_subjects = sub["sub_char"].shape[0]

        for i in range(n_subjects):
            print(f"Processing subject {i+1}/{n_subjects}")

            subject = {}

            # ---- subject metadata ----
            subject["sub_char"] = deref(f, f[sub["sub_char"][i, 0]])
            #subject["meas_char"] = deref(f, f[sub["meas_char"][i, 0]])
            #subject["events"] = deref(f, f[sub["events"][i, 0]])

            # ---- segmented data ----
            subject["LsideSegm"] = {
                "Bside": deref(f, f[sub["LsideSegm_BsideData"][i, 0]]),
                "Lside": deref(f, f[sub["LsideSegm_LsideData"][i, 0]]),
                "Rside": deref(f, f[sub["LsideSegm_RsideData"][i, 0]])
            }
            """
            subject["RsideSegm"] = {
                "Bside": deref(f, f[sub["RsideSegm_BsideData"][i, 0]]),
                "Lside": deref(f, f[sub["RsideSegm_LsideData"][i, 0]])
                #"Rside": deref(f, f[sub["RsideSegm_RsideData"][i, 0]]),
            }
            """

            #subjects.append(subject)
            out_path = f"data/subject_{i:03d}.pkl"
            import pickle
            with open(out_path, "wb") as fp:
                pickle.dump(subject, fp, protocol=pickle.HIGHEST_PROTOCOL)

            del subject  # free memory explicitly
        

    #return subjects


# -------- RUN ----------
load_all_subjects(file_path)




"""
import pickle

output_path = "able_body_pkl_nature2.pkl"

with open(output_path, "wb") as f:
    pickle.dump(subjects, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved to {output_path}")
"""
