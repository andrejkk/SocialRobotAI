#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import torch


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


for dirname, _, filenames in os.walk('dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# %%



src_dir = "/kaggle/input/opportunity-adl"
user_num = 4
adl_file_num = 5
files_dir = {}
for user_idx in range(1, user_num+1):
    user_data_path = []
    for file_idx in range(1, adl_file_num+1):
        file_name = f"S{user_idx}-ADL{file_idx}.dat"
        file_path = os.path.join(src_dir, file_name)
        user_data_path.append(file_path)
    files_dir[str(user_idx)] = user_data_path

files_dir
# %%

def get_cleaned_user_data(file_pth, user_idx):
    """去除不用传感器模态，去除活动转换数据，插值补全NaN数据"""
    invalid_feature = np.arange( 46, 50 )  # BACK Quaternion
    invalid_feature = np.concatenate( [invalid_feature, np.arange(34, 37)] )  # RH_acc
    invalid_feature = np.concatenate( [invalid_feature, np.arange(59, 63)] )  # RUA Quaternion
    invalid_feature = np.concatenate( [invalid_feature, np.arange(72, 76)] )  # RLA
    invalid_feature = np.concatenate( [invalid_feature, np.arange(85, 89)] )  # LUA
    invalid_feature = np.concatenate( [invalid_feature, np.arange(99, 102)] )  # LLA
    invalid_feature = np.concatenate( [invalid_feature, np.arange(117, 118)] )  # L-SHOE Compass
    invalid_feature = np.concatenate( [invalid_feature, np.arange(133, 134)] )  # R-SHOE Compass
    invalid_feature = np.concatenate( [invalid_feature, np.arange(134, 244)] )  # environment sensor
    invalid_feature = np.concatenate( [invalid_feature, np.arange(245, 250)] )  # LL, ML level label
    drop_columns = invalid_feature
#     print(drop_columns, len(drop_columns))
    raw_data = np.loadtxt(file_pth)
#     print(raw_data.shape)
    used_data = np.delete(raw_data, drop_columns, axis=1)
    print(used_data.shape)    
    
    used_columns = ["MILLISEC",
                    "acc_RKN_upper_accX","acc_RKN_upper_accY","acc_RKN_upper_accZ",
                    "acc_HIP_accX","acc_HIP_accY","acc_HIP_accZ",
                    "acc_LUA_upper_accX","acc_LUA_upper_accY","acc_LUA_upper_accZ",
                    "acc_RUA_lower_accX","acc_RUA_lower_accY","acc_RUA_lower_accZ",
                    "acc_LH_accX","acc_LH_accY","acc_LH_accZ",
                    "acc_BACK_accX","acc_BACK_accY","acc_BACK_accZ",
                    "acc_RKN_lower_accX","acc_RKN_lower_accY","acc_RKN_lower_accZ",
                    "acc_RWR_accX","acc_RWR_accY","acc_RWR_accZ",
                    "acc_RUA_upper_accX","acc_RUA_upper_accY","acc_RUA_upper_accZ",
                    "acc_LUA_lower_accX","acc_LUA_lower_accY","acc_LUA_lower_accZ",
                    "acc_LWR_accX","acc_LWR_accY","acc_LWR_accZ",
#                     "acc_RH_accX","acc_RH_accY","acc_RH_accZ",
                    "imu_BACK_accX","imu_BACK_accY","imu_BACK_accZ",
                    "imu_BACK_gyroX","imu_BACK_gyroY","imu_BACK_gyroZ",
                    "imu_BACK_magneticX","imu_BACK_magneticY","imu_BACK_magneticZ",
                    "imu_RUA_accX","imu_RUA_accY","imu_RUA_accZ",
                    "imu_RUA_gyroX","imu_RUA_gyroY","imu_RUA_gyroZ",
                    "imu_RUA_magneticX","imu_RUA_magneticY","imu_RUA_magneticZ",
                    "imu_RLA_accX","imu_RLA_accY","imu_RLA_accZ",
                    "imu_RLA_gyroX","imu_RLA_gyroY","imu_RLA_gyroZ",
                    "imu_RLA_magneticX","imu_RLA_magneticY","imu_RLA_magneticZ",
                    "imu_LUA_accX","imu_LUA_accY","imu_LUA_accZ",
                    "imu_LUA_gyroX","imu_LUA_gyroY","imu_LUA_gyroZ",
                    "imu_LUA_magneticX","imu_LUA_magneticY","imu_LUA_magneticZ",
                    "imu_LLA_accX","imu_LLA_accY","imu_LLA_accZ",
                    "imu_LLA_gyroX","imu_LLA_gyroY","imu_LLA_gyroZ",
                    "imu_LLA_magneticX","imu_LLA_magneticY","imu_LLA_magneticZ",
                    "imu_L-SHOE_EuX","imu_L-SHOE_EuY","imu_L-SHOE_EuZ",
                    "imu_L-SHOE_Nav_Ax","imu_L-SHOE_Nav_Ay","imu_L-SHOE_Nav_Az",
                    "imu_L-SHOE_Body_Ax","imu_L-SHOE_Body_Ay","imu_L-SHOE_Body_Az",
                    "imu_L-SHOE_AngVelBodyFrameX","imu_L-SHOE_AngVelBodyFrameY","imu_L-SHOE_AngVelBodyFrameZ",
                    "imu_L-SHOE_AngVelNavFrameX","imu_L-SHOE_AngVelNavFrameY","imu_L-SHOE_AngVelNavFrameZ",
                    "imu_R-SHOE_EuX","imu_R-SHOE_EuY","imu_R-SHOE_EuZ",
                    "imu_R-SHOE_Nav_Ax","imu_R-SHOE_Nav_Ay","imu_R-SHOE_Nav_Az",
                    "imu_R-SHOE_Body_Ax","imu_R-SHOE_Body_Ay","imu_R-SHOE_Body_Az",
                    "imu_R-SHOE_AngVelBodyFrameX","imu_R-SHOE_AngVelBodyFrameY","imu_R-SHOE_AngVelBodyFrameZ",
                    "imu_R-SHOE_AngVelNavFrameX","imu_R-SHOE_AngVelNavFrameY","imu_R-SHOE_AngVelNavFrameZ",
                    "Locomotion",
                    "HL_Activity"]
    used_data = pd.DataFrame(used_data, columns=used_columns)
#     print(used_data.shape)
    used_data = used_data[used_data['HL_Activity'] != 0]  # 活动转换数据标签为0，丢弃
#     print(used_data.shape)

    used_data['HL_Activity'][used_data['HL_Activity']==101] = 0  # Relaxing
    used_data['HL_Activity'][used_data['HL_Activity']==102] = 1  # Coffee time
    used_data['HL_Activity'][used_data['HL_Activity']==103] = 2  # Early morning
    used_data['HL_Activity'][used_data['HL_Activity']==104] = 3  # Cleanup
    used_data['HL_Activity'][used_data['HL_Activity']==105] = 4  # Sandwich time
    
#     print(used_data.shape)
    used_data = used_data.interpolate()
#     print(used_data.shape)

    # 查看Nan数据所在位置
    pos = used_data.isnull().stack()[lambda x:x].index.tolist()
#     print(pos)
    
    used_data = used_data.dropna(axis=0)
    print(used_data.shape)
    return used_data
# %%

warnings.filterwarnings('ignore')

scaler = StandardScaler()

def sliding_window(time_series, width, step, order='F'):
    w = np.hstack([time_series[i:1 + i - width or None:step] for i in range(0, width)])
    result = w.reshape((int(len(w) / width), width), order='F')
    if order == 'F':
        return result
    else:
        return np.ascontiguousarray(result)
    
def calc_normalization(data):
    num_instances, num_time_steps, num_features = data.shape
    data = np.reshape(data, (num_instances, -1))
    scaler.fit(data)
#     mean, std = (np.array([np.mean(x) for x in X_train], dtype=np.float32), np.array([np.std(x) for x in X_train], dtype=np.float32))
    return scaler
    
def apply_normalization(data, scaler):
#     scaler = StandardScaler()
    num_instances, num_time_steps, num_features = data.shape
    data = np.reshape(data, (num_instances, -1))
    norm_data = scaler.transform(data)
#     debug_here()
#     data = (data - mean) / (std + 1e-5)
    norm_data[np.isnan(norm_data)] = 0
    norm_data = np.reshape(norm_data, (num_instances, num_time_steps, num_features))
    return norm_data
# %%


src_dir = "dataset"
user_num = 4
adl_file_num = 5

seq_length = 300
shifting_step = 30
channel_num = 3*36
used_channels = [
    "acc_RKN_upper_accX","acc_RKN_upper_accY","acc_RKN_upper_accZ",
    "acc_HIP_accX","acc_HIP_accY","acc_HIP_accZ",
    "acc_LUA_upper_accX","acc_LUA_upper_accY","acc_LUA_upper_accZ",
    "acc_RUA_lower_accX","acc_RUA_lower_accY","acc_RUA_lower_accZ",
    "acc_LH_accX","acc_LH_accY","acc_LH_accZ",
    "acc_BACK_accX","acc_BACK_accY","acc_BACK_accZ",
    "acc_RKN_lower_accX","acc_RKN_lower_accY","acc_RKN_lower_accZ",
    "acc_RWR_accX","acc_RWR_accY","acc_RWR_accZ",
    "acc_RUA_upper_accX","acc_RUA_upper_accY","acc_RUA_upper_accZ",
    "acc_LUA_lower_accX","acc_LUA_lower_accY","acc_LUA_lower_accZ",
    "acc_LWR_accX","acc_LWR_accY","acc_LWR_accZ",
    "imu_BACK_accX","imu_BACK_accY","imu_BACK_accZ",
    "imu_BACK_gyroX","imu_BACK_gyroY","imu_BACK_gyroZ",
    "imu_BACK_magneticX","imu_BACK_magneticY","imu_BACK_magneticZ",
    "imu_RUA_accX","imu_RUA_accY","imu_RUA_accZ",
    "imu_RUA_gyroX","imu_RUA_gyroY","imu_RUA_gyroZ",
    "imu_RUA_magneticX","imu_RUA_magneticY","imu_RUA_magneticZ",
    "imu_RLA_accX","imu_RLA_accY","imu_RLA_accZ",
    "imu_RLA_gyroX","imu_RLA_gyroY","imu_RLA_gyroZ",
    "imu_RLA_magneticX","imu_RLA_magneticY","imu_RLA_magneticZ",
    "imu_LUA_accX","imu_LUA_accY","imu_LUA_accZ",
    "imu_LUA_gyroX","imu_LUA_gyroY","imu_LUA_gyroZ",
    "imu_LUA_magneticX","imu_LUA_magneticY","imu_LUA_magneticZ",
    "imu_LLA_accX","imu_LLA_accY","imu_LLA_accZ",
    "imu_LLA_gyroX","imu_LLA_gyroY","imu_LLA_gyroZ",
    "imu_LLA_magneticX","imu_LLA_magneticY","imu_LLA_magneticZ",
    "imu_L-SHOE_EuX","imu_L-SHOE_EuY","imu_L-SHOE_EuZ",
    "imu_L-SHOE_Nav_Ax","imu_L-SHOE_Nav_Ay","imu_L-SHOE_Nav_Az",
    "imu_L-SHOE_Body_Ax","imu_L-SHOE_Body_Ay","imu_L-SHOE_Body_Az",
    "imu_L-SHOE_AngVelBodyFrameX","imu_L-SHOE_AngVelBodyFrameY","imu_L-SHOE_AngVelBodyFrameZ",
    "imu_L-SHOE_AngVelNavFrameX","imu_L-SHOE_AngVelNavFrameY","imu_L-SHOE_AngVelNavFrameZ",
    "imu_R-SHOE_EuX","imu_R-SHOE_EuY","imu_R-SHOE_EuZ",
    "imu_R-SHOE_Nav_Ax","imu_R-SHOE_Nav_Ay","imu_R-SHOE_Nav_Az",
    "imu_R-SHOE_Body_Ax","imu_R-SHOE_Body_Ay","imu_R-SHOE_Body_Az",
    "imu_R-SHOE_AngVelBodyFrameX","imu_R-SHOE_AngVelBodyFrameY","imu_R-SHOE_AngVelBodyFrameZ",
    "imu_R-SHOE_AngVelNavFrameX","imu_R-SHOE_AngVelNavFrameY","imu_R-SHOE_AngVelNavFrameZ",
]


for user_idx in range(1, user_num+1):
    user_data, user_labels = [], []
    for file_idx in range(1, adl_file_num+1):
        # gen src_data path
        file_name = f"S{user_idx}-ADL{file_idx}.dat"
        file_path = os.path.join(src_dir, file_name)
        
        # load cleaned data
        used_data = get_cleaned_user_data(file_path, user_idx)
        
        # split data by label
        for act_id, act_data in used_data.groupby('HL_Activity'):
#             print(act_id)
#             print(act_data.shape)
            sample_cnt = int((act_data.shape[0]-seq_length)//shifting_step + 1)
            if sample_cnt < 2:
                print(f"user {user_index} has only {act_data.shape[0]} samplings, drop\n")
                continue
            data_shape = (sample_cnt, seq_length, channel_num)  # (N, 300, 3*36)
            act_sliced_data = np.empty(data_shape)  
            channl_idx = 0
            for channel_name in used_channels:
                channel_data = act_data[channel_name]
                act_sliced_data[:,:,channl_idx] = sliding_window(channel_data.values, seq_length, shifting_step, 'T')
                channl_idx += 1

            # append label data 
            user_data.append(act_sliced_data)
            # gen labels
            class_labels = np.empty(act_sliced_data.shape[0])
            actual_label = int(act_id)
            class_labels.fill(actual_label)
            user_labels.append(class_labels.astype(int))
            
    # data and labels for each users 
    array_user_data= np.concatenate(user_data, axis=0)
    array_user_labels= np.concatenate(user_labels, axis=0)
    # print(user_idx, array_user_data.shape, array_user_labels.shape)
    
    # Stratified train, validation, test split of the data 
    X_train, X_test, y_train, y_test = train_test_split(array_user_data, array_user_labels,  stratify=array_user_labels,  test_size=0.3,random_state=1)
    # print(X_train.shape)
    # print(y_train.shape)

    # Data normalization 
    # Calculate mean and standard deviation based on train
    scaler = calc_normalization(X_train)

    # Apply normalization 
    X_train = apply_normalization(X_train,scaler)
    X_test = apply_normalization(X_test,scaler)

    print(f"user: {user_idx}")
    print(f"train data: {X_train.shape}, train label: {y_train.shape}")
    print(f"test data: {X_test.shape}, test label: {y_test.shape}\n")
    
    # prepare samples
    train_data = {'samples':X_train, 'labels':y_train}
    test_data  = {'samples':X_test, 'labels':y_test}

    os.makedirs(f'prepared-data', exist_ok=True)
    torch.save(train_data, f'prepared-data/train_{user_idx}.pt')
    # torch.save(val_data,  f'HHAR_user_data/val_{user_name}.pt')
    torch.save(test_data, f'prepared-data/test_{user_idx}.pt')
# %%
