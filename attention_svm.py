from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import joblib

# def load_train_data():
#     try:
#         df = pd.read_csv('train/attention_detection_dataset_v1.csv',
#                      delimiter=',', encoding='ISO-8859-1')
#     except FileNotFoundError:
#         print("napaka pri branju datoteke, preveri pot do datoteke")
#         return None

#     # odstrani stolpce ki nerabis
#     df.drop(['phone','phone_x', 'phone_y','phone_w', 'phone_h','phone_con', 'no_of_face', 'no_of_hand', 'face_con', 'pose'], axis=1, inplace=True)
#     print("TRAIN: ")
#     print(df.head())
#     return df

# def load_test_data():
#     try:
#         df = pd.read_csv('train/attention_detection_dataset_v2.csv',
#                      delimiter=',', encoding='ISO-8859-1')
#     except FileNotFoundError:
#         print("napaka pri branju datoteke, preveri pot do datoteke")
#         return None

#     # odstrani stolpce ki nerabis
#     df.drop(['phone','phone_x', 'phone_y','phone_w', 'phone_h','phone_con', 'no_of_face', 'no_of_hand', 'face_con', 'pose'], axis=1, inplace=True)
#     print("TEST: ")
#     print(df.head())    
#     return df

# train_df = load_train_data()
# test_df = load_test_data()



############kaj pa ce prek tega probam?
# def face_landmarks_to_features(face_landmarks) -> np.ndarray:
#     """Convert MediaPipe FaceMesh landmarks to a fixed-length feature vector.

#     The goal is to encode head pose and eye-gaze-related cues that an attention
#     classifier can use. This is generic enough for a webcam or video pipeline.
#     """
#     if face_landmarks is None or len(face_landmarks.landmark) == 0:
#         return np.zeros(15, dtype=np.float32)

#     pts = np.array(
#         [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
#         dtype=np.float32,
#     )

#     def p(idx):
#         return pts[idx]

#     # --- face geometry ---
#     face_center = pts[[0, 1, 33, 61, 291, 199]].mean(axis=0)
#     nose = p(1)
#     chin = p(152)
#     left_eye_corner = p(33)
#     right_eye_corner = p(263)
#     left_ear = p(234)
#     right_ear = p(454)

#     face_width = np.linalg.norm(right_ear - left_ear)
#     face_height = np.linalg.norm(chin - nose)
#     eye_distance = np.linalg.norm(right_eye_corner - left_eye_corner)

#     # --- head pose approximations from geometry ---
#     # Yaw: horizontal offset between eye corners
#     yaw = (right_eye_corner[0] - left_eye_corner[0]) / (eye_distance + 1e-8)
#     # Pitch: vertical difference between nose and chin
#     pitch = (nose[1] - chin[1]) / (face_height + 1e-8)
#     # Roll: relative orientation from ear-to-ear line vs vertical axis
#     ear_line = right_ear - left_ear
#     roll = np.arctan2(ear_line[1], ear_line[0])  # rough 2D head roll estimate

#     # --- eye features ---
#     left_eye_idx = [33, 133, 159, 145, 153, 263]
#     right_eye_idx = [362, 382, 381, 380, 374, 386]

#     left_eye = pts[left_eye_idx]
#     right_eye = pts[right_eye_idx]
#     left_eye_center = left_eye.mean(axis=0)
#     right_eye_center = right_eye.mean(axis=0)

#     def eye_aspect_ratio(eye_points):
#         top = eye_points[1]
#         bottom = eye_points[4]
#         left = eye_points[0]
#         right = eye_points[3]
#         vertical = np.linalg.norm(top - bottom)
#         horizontal = np.linalg.norm(left - right)
#         return vertical / (horizontal + 1e-8)

#     left_ear_ratio = eye_aspect_ratio(left_eye)
#     right_ear_ratio = eye_aspect_ratio(right_eye)

#     # Eye gaze estimate: relative offset between left and right eye centers
#     gaze_x = right_eye_center[0] - left_eye_center[0]
#     gaze_y = right_eye_center[1] - left_eye_center[1]
#     gaze_z = right_eye_center[2] - left_eye_center[2]

#     # Additional spatial cues to help attention detection
#     head_center_offset = face_center[0] - 0.5
#     head_vertical = face_center[1] - 0.5

#     feature_vector = np.array(
#         [
#             yaw,
#             pitch,
#             roll,
#             gaze_x,
#             gaze_y,
#             gaze_z,
#             left_ear_ratio,
#             right_ear_ratio,
#             face_width,
#             face_height,
#             eye_distance,
#             head_center_offset,
#             head_vertical,
#             left_eye_center[0] - 0.5,
#             right_eye_center[0] - 0.5,
#         ],
#         dtype=np.float32,
#     )

#     return feature_vector



def load_data():
    try:
        df = pd.read_csv('trainv3/attention_detection_dataset_v3.csv',
                     delimiter=',', encoding='ISO-8859-1')
    except FileNotFoundError:
        print("napaka pri branju datoteke, preveri pot do datoteke")
        return None

    # odstrani stolpce ki nerabis
    # df.drop(['face_present','no_of_face','hand_count','left_hand_x','left_hand_y','right_hand_x','right_hand_y','hand_obj_interaction','head_pose','phone_present', 'phone_loc_x', 'phone_loc_y', 'phone_conf', 'gaze_on_screen', 'gaze_direction','gazePoint_x','gazePoint_y','pupil_left_x','pupil_left_y','pupil_right_x','pupil_right_y', 'head_pitch','head_yaw','head_roll']
    # ,axis=1, inplace=True)

    #ta vsebuje še pitch yaw pa roll
    #dal ven 'face_present',
    df.drop(['no_of_face','hand_count','left_hand_x','left_hand_y','right_hand_x','right_hand_y','hand_obj_interaction','head_pose','phone_present', 'phone_loc_x', 'phone_loc_y', 'phone_conf', 'gaze_on_screen', 'gaze_direction','gazePoint_x','gazePoint_y', 'pupil_left_x','pupil_left_y','pupil_right_x','pupil_right_y']
    ,axis=1, inplace=True)
    # print(df.head())    
    # print(df[[
    #     "head_pitch",
    #     "head_yaw",
    #     "head_roll"
    # ]].describe())
    return df

df = load_data()

#tole uporabi ce bos dubu tud pupil coords pri mediapipe
#izracunaj NORMALIZED TO FACE(W,H) PUPIL COORDS for gaze estimation
# df['pupil_left_x'] = df['pupil_left_x'] / df['face_w']
# df['pupil_left_y'] = df['pupil_left_y'] / df['face_h']
# df['pupil_right_x'] = df['pupil_right_x'] / df['face_w']
# df['pupil_right_y'] = df['pupil_right_y'] / df['face_h']

#mouth center baje nemorem dobit sam iz mouth_x pa mouth_y, ker je to samo ena tocka, ne pa sredisce med left_mouth in right_mouth, zato to ne pride v upostev:
# df["mouth_center_x"] = (df["mouth_x"] + df["mouth_y"]) / 2.0 #nemors uporabit
# df["mouth_center_y"] = (df["left_mouth_y"] + df["right_mouth_y"]) / 2.0 #nemors uporabit

# print("PUPIL COORDS NORMALIZED TO FACE SIZE:")
# print(df[[
#     "pupil_left_x",
#     "pupil_left_y",
#     "pupil_right_x",
#     "pupil_right_y"
# ]].describe())

################ ZGRADI FEATURJE OCES, UST IN NOSA ####################
#najprej normalizirat koordinate za eyes in mouth na velikost obraza, da dobim relativne koordinate glede na velikost obraza

#TODO: a iz teh raj nov df zgradim al updateam izvornega??
df = df[df['face_present'] == 1].copy()
# #1. normalizacija
# #-----OCESA------
# #LEVO OKO, x normed to w in y normed to h, dobimo norm_left_eye(x,y)
# # df["left_eye_x"] = df["left_eye_x"] / df["face_w"]
# # df["left_eye_y"] = df["left_eye_y"] / df["face_h"]
# left_eye_x = df["left_eye_x"] / df["face_w"]
# left_eye_y = df["left_eye_y"] / df["face_h"]

# #al moram celo:
# left_eye_x_norm = (
#     df["left_eye_x"] - df["face_x"]
# ) / df["face_w"]

# left_eye_y_norm = (
#     df["left_eye_y"] - df["face_y"]
# ) / df["face_h"]

#DESNO OKO, x normed to w in y normed to h, dobimo norm_right_eye(x,y)
# right_eye_x = (df["right_eye_x"] / df["face_w"]
# right_eye_y = (df["right_eye_y"] / df["face_h"]

#isto tu odštet face x pa face y
# right_eye_x_norm = (
#     df["right_eye_x"] - df["face_x"]
# ) / df["face_w"]

# right_eye_y_norm = (
#     df["right_eye_y"] - df["face_y"]
# ) / df["face_h"]


#SREDISCNA TOCKA OCES
# eye_center_x = (df["right_eye_x"] + df["left_eye_x"]) / 2.0
# eye_center_y = (df["right_eye_y"] + df["left_eye_y"]) / 2.0
# eye_center = np.array(eye_center_x, eye_center_y)
# #-----OCESA------

# #-----USTA-------
# #iz le (x,y) ene tocke ust nemorem dobit sredisca, dobimo norm_mouth(x,y)
# mouth_x_norm = (df["mouth_x"] - df["face_x"]) / df["face_w"] #x -> po sirini
# mouth_y_norm = (df["mouth_y"] - df["face_y"]) / df["face_h"] #y -> po visini
# mouth_center = (mouth_x_norm + mouth_y_norm ) / 2

# #nimam framea v csv tkoda to pomoje negre?
# # df["mouth_x"] = df["mouth_x"] / df["frame_width"]
# # df["mouth_y"] = df["mouth_y"] / df["frame_height"]
# #-----USTA-------

# #-----NOS--------
# #NOS da dobim norm_nose_tip(x,y) isto naredim kot za usta
# nose_tip_x_norm = (df["nose_tip_x"] - df["face_x"]) / df["face_w"]
# nose_tip_y_norm = (df["nose_tip_y"] - df["face_y"]) / df["face_h"]
# #-----NOS--------
#######################BAJE NAJ RAJE NORMALIZIRAM Z IOD KOKR PA FACE W PA H


####ZA DISTANCES ZA VSAK FEATURE VEDNO POD SQUARE ROOT: 
#kjer je interocular dist = (right_eye_x - right_eye_y)^2 + (left_eye_x - left_eye_y)^2 oz koren( (x2-x1)^2 + (y2-y1)^2 )

#2. relativne razdalje med:
    #- levim in desnim ocesom = interocular preko evklida,ker nimam koordinat pupilov za IPD
    #- razdalje od sredisca oci in ust, ce je to viable metric dist= center_eye <-> mouth
# interocular_eye_dist = np.sqrt(
#     (df["left_eye_x"] - df["right_eye_x"]) ** 2 +
#     (df["left_eye_y"] - df["right_eye_y"]) ** 2
# )
# #oz direkt preko
# eye_distance = np.linalg.norm([
#     df["left_eye_x"] - df["right_eye_x"], 
#     df["left_eye_y"] - df["right_eye_y"]]
# )

# left_eye_mouth_distance = np.sqrt(
#     (df["mouth_x"] - df["left_eye_x"]) ** 2 + 
#     (df["mouth_y"] - df["left_eye_y"]) ** 2
# )

# right_eye_mouth_distance = np.sqrt(
#     (df["mouth_x"] - df["right_eye_x"]) ** 2 + 
#     (df["mouth_y"] - df["right_eye_y"]) ** 2
# )

# nose_mouth_distance = np.sqrt(
#     (df["mouth_x"] - df["nose_tip_x"]) ** 2 +
#     (df["mouth_y"] - df["nose_tip_y"]) ** 2
# )

#tega nemorem ker nimam leve_mouth x,y in desne_mouth x,y
# df["eye_mouth_dist"] = np.sqrt(
#     (df["eye_center_x"] - df["mouth_center_x"]) ** 2 +
#     (df["eye_center_y"] - df["mouth_center_y"]) ** 2
# )


#kar je isto kot: calculate IOD interocular distance
# df["IOD"] = np.sqrt(
#     (df["left_eye_x"] - df["right_eye_x"]) ** 2 + 
#     (df["left_eye_y"] - df["right_eye_y"])** 2
# )
#df["IOD"] = interocular_eye_dist
#df["interocular_eye_dist"] = interocular_eye_dist



#iz cesar lah dobim normalizirane koordinate za sredisce med levim in desnim okom, pa tudi za sredisce med usti
#sredisce oci po x in sredisce po y
# df["eye_center_x"] = (df["left_eye_x"] + df["right_eye_x"]) / 2.0 #to je sigurno ane
# df["eye_center_y"] = (df["left_eye_y"] + df["right_eye_y"]) / 2.0
#####imam prej napisano gor za srediscno tocko oces

#sredisce med mouth_x in mouth_y
# df["mouth_center_x"] = (df["left_mouth_x"] + df["right_mouth_x"]) / 2.0
# df["mouth_center_y"] = (df["left_mouth_y"] + df["right_mouth_y"]) / 2.0

#LIN ALG NORM JE ISTO KOT EVKLID LOH BI UPORABU TO NAMESTO PISANJA FORMULE RAZDALJE
#relativna razdalja od uci do ust

#nimam pupilov v pose landmarku, zato ne morem izracunat IPD, ampak lahko izracunam IOD (interocular distance) med levim in desnim okom
# #calculate IPD interpupillary distance
# df["IPD"] = np.sqrt(
#     (df["pupil_left_x"] - df["pupil_right_x"]) **
#     2 + (df["pupil_left_y"] - df["pupil_right_y"]) ** 2
# )

#kako pol to uporabim iz dobljenih koordinat?
# eps = 1e-6
# df["IPD_ratio"] = df["IPD"] / (df["IOD"] + eps)

# eye_distance = np.maximum(eye_distance, eps)

eye_mid_x = (df['left_eye_x'] + df['right_eye_x']) / 2
eye_mid_y = (df['left_eye_y'] + df['right_eye_y']) / 2
iod = np.sqrt((df['right_eye_x'] - df['left_eye_x'])**2 + (df['right_eye_y'] - df['left_eye_y'])**2)

# df['left_eye_x_norm']  = (df['left_eye_x']  - eye_mid_x) / iod
# df['left_eye_y_norm']  = (df['left_eye_y']  - eye_mid_y) / iod
# df['right_eye_x_norm'] = (df['right_eye_x'] - eye_mid_x) / iod
# df['right_eye_y_norm'] = (df['right_eye_y'] - eye_mid_y) / iod
# df['nose_tip_x_norm']  = (df['nose_tip_x']  - eye_mid_x) / iod
# df['nose_tip_y_norm']  = (df['nose_tip_y']  - eye_mid_y) / iod
# df['mouth_x_norm']     = (df['mouth_x']     - eye_mid_x) / iod
# df['mouth_y_norm']     = (df['mouth_y']     - eye_mid_y) / iod

# pure distances - no subtraction needed, per the earlier rule (it cancels out algebraically)
#TODO a te rabim od clauda?? probably dobro da mam??
# df['nose_mouth_dist']      = np.sqrt((df['mouth_x']-df['nose_tip_x'])**2 + (df['mouth_y']-df['nose_tip_y'])**2) / iod
# df['left_eye_mouth_dist']  = np.sqrt((df['mouth_x']-df['left_eye_x'])**2 + (df['mouth_y']-df['left_eye_y'])**2) / iod
# df['right_eye_mouth_dist'] = np.sqrt((df['mouth_x']-df['right_eye_x'])**2 + (df['mouth_y']-df['right_eye_y'])**2) / iod

    
left_eye_x_norm = (df['left_eye_x'] - eye_mid_x) / iod
left_eye_y_norm = (df['left_eye_y'] - eye_mid_y) / iod
right_eye_x_norm = (df['right_eye_x'] - eye_mid_x) / iod
right_eye_y_norm = (df['right_eye_y'] - eye_mid_y) / iod
nose_tip_x_norm =(df['nose_tip_x'] - eye_mid_x) / iod
nose_tip_y_norm = (df['nose_tip_y'] - eye_mid_y) / iod
mouth_x_norm = (df['mouth_x'] - eye_mid_x) / iod
mouth_y_norm = (df['mouth_y'] - eye_mid_y) / iod


new_df_iod = pd.DataFrame({
    "left_eye_x": left_eye_x_norm,
    "left_eye_y": left_eye_y_norm,
    "right_eye_x": right_eye_x_norm,
    "right_eye_y": right_eye_y_norm,
    
    "nose_tip_x": nose_tip_x_norm,
    "nose_tip_y": nose_tip_y_norm,

    "mouth_x": mouth_x_norm,
    "mouth_y": mouth_y_norm,
    #"mouth_center": mouth_center,
    # "head_pitch": df["head_pitch"],
    # "head_yaw": df["head_yaw"],
    # "head_roll": df["head_roll"],
    "label": df["label"]
}, index=df.index)

print(new_df_iod)


#####brez IOD:
# new_df = pd.DataFrame({
#     "left_eye_x": left_eye_x_norm,
#     "left_eye_y": left_eye_y_norm,
#     "right_eye_x": right_eye_x_norm,
#     "right_eye_y": right_eye_y_norm,
    
#     "nose_tip_x": nose_tip_x_norm,
#     "nose_tip_y": nose_tip_y_norm,

#     "mouth_x": mouth_x_norm,
#     "mouth_y": mouth_y_norm,
#     #"mouth_center": mouth_center,
#     "head_pitch": df["head_pitch"],
#     "head_yaw": df["head_yaw"],
#     "head_roll": df["head_roll"],
#     "label": df["label"]
# }, index=df.index)

#print(new_df)
#print(new_df_iod)

train_df, test_df = train_test_split(new_df_iod, test_size=0.2, random_state=42, stratify=df['label'])

#binarni target column za attentiveness je label
target_column = 'label'

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

print("\nSVM features:")
for i, column in enumerate(X_train.columns):
    print(i, column)


X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))
print("Number of features:", X_train.shape[1])


# numeric_features = [
#     'face_x',
#     'face_y',
#     'face_w',
#     'face_h',
#     #'face_con',
#     'pose_x',
#     'pose_y'
# ]
#categorical_features = ['pose']
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ]
# )

model = make_pipeline(
    #preprocessor,
    StandardScaler(),
    SVC(
        kernel="rbf",
        probability=True,
        C=1.0, 
        gamma='scale', 
        random_state=42,
        #class_weight="balanced"
    )
)
# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("Training SVM...")

model.fit(X_train, y_train)

print("Training finished, saving model...")

joblib.dump(model, "attention_svm.joblib")



# --------------------------------------------------
# TEST
# --------------------------------------------------
y_pred = model.predict(X_test)
# --------------------------------------------------
# RESULTS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("RESULTS")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
# --------------------------------------------------
# CONFUSION MATRIX PLOT
# --------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("SVM Attention Classification")
plt.show()