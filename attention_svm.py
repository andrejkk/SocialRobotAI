from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

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

#from solvepnp import calculate_head_pose

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
# print(df[
#     [
#         "face_x", "face_y", "face_w", "face_h",
#         "left_eye_x", "left_eye_y",
#         "right_eye_x", "right_eye_y",
#         "nose_tip_x", "nose_tip_y",
#         "mouth_x", "mouth_y"
#     ]
# ].describe())

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


# # USTA
# #iz le (x,y) ene tocke ust nemorem dobit sredisca, dobimo norm_mouth(x,y)
# mouth_x_norm = (df["mouth_x"] - df["face_x"]) / df["face_w"] #x -> po sirini
# mouth_y_norm = (df["mouth_y"] - df["face_y"]) / df["face_h"] #y -> po visini
# mouth_center = (mouth_x_norm + mouth_y_norm ) / 2

# #nimam framea v csv tkoda to pomoje negre?
# # df["mouth_x"] = df["mouth_x"] / df["frame_width"]
# # df["mouth_y"] = df["mouth_y"] / df["frame_height"]
# 

# # NOS
# #NOS da dobim norm_nose_tip(x,y) isto naredim kot za usta
# nose_tip_x_norm = (df["nose_tip_x"] - df["face_x"]) / df["face_w"]
# nose_tip_y_norm = (df["nose_tip_y"] - df["face_y"]) / df["face_h"]
#
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


#kar je isto kot:IOD distanca
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
# IPD
# df["IPD"] = np.sqrt(
#     (df["pupil_left_x"] - df["pupil_right_x"]) **
#     2 + (df["pupil_left_y"] - df["pupil_right_y"]) ** 2
# )


#TODO probably dobro da mam??
# df['nose_mouth_dist']      = np.sqrt((df['mouth_x']-df['nose_tip_x'])**2 + (df['mouth_y']-df['nose_tip_y'])**2) / iod
# df['left_eye_mouth_dist']  = np.sqrt((df['mouth_x']-df['left_eye_x'])**2 + (df['mouth_y']-df['left_eye_y'])**2) / iod
# df['right_eye_mouth_dist'] = np.sqrt((df['mouth_x']-df['right_eye_x'])**2 + (df['mouth_y']-df['right_eye_y'])**2) / iod

iod = np.sqrt((df['right_eye_x'] - df['left_eye_x'])**2 + (df['right_eye_y'] - df['left_eye_y'])**2)
eye_mid_x = (df['left_eye_x'] + df['right_eye_x']) / 2
eye_mid_y = (df['left_eye_y'] + df['right_eye_y']) / 2

eye_nose_offset_x = (df["nose_tip_x"] - eye_mid_x) / iod
eye_nose_offset_y = (df["nose_tip_y"] - eye_mid_y) / iod

left_eye_x_norm = (df['left_eye_x'] - eye_mid_x) / iod
left_eye_y_norm = (df['left_eye_y'] - eye_mid_y) / iod
right_eye_x_norm = (df['right_eye_x'] - eye_mid_x) / iod
right_eye_y_norm = (df['right_eye_y'] - eye_mid_y) / iod
nose_tip_x_norm =(df['nose_tip_x'] - eye_mid_x) / iod
nose_tip_y_norm = (df['nose_tip_y'] - eye_mid_y) / iod
mouth_x_norm = (df['mouth_x'] - eye_mid_x) / iod
mouth_y_norm = (df['mouth_y'] - eye_mid_y) / iod

# racunanje premikov glave iz oci in nosa
roll_proxy = np.arctan2(df['right_eye_y'] - df['left_eye_y'], df['right_eye_x'] - df['left_eye_x'])
yaw_proxy = np.arctan2(nose_tip_x_norm, 1.0)
pitch_proxy = np.arctan2(nose_tip_y_norm, 1.0)

# yaw_proxy = np.arctan2(
#     df["nose_tip_x"] - eye_mid_x,
#     iod
# )
# pitch_proxy = np.arctan2(
#     eye_mid_y - df["nose_tip_y"],
#     iod
# )

# yaw_proxy = (df['nose_tip_x'] - eye_mid_x) / iod 
# pitch_proxy = (df['nose_tip_y'] - eye_mid_y) / iod

# pitch = np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2])
# yaw = np.arctan2(nose_tip[0] - chin[0], nose_tip[2] - chin[2])

# gaze_x = right_eye_center[0] - left_eye_center[0]
# gaze_y = right_eye_center[1] - left_eye_center[1]
# gaze_z = right_eye_center[2] - left_eye_center[2]

gaze_x = eye_mid_x - eye_mid_x
gaze_y = eye_mid_y - eye_mid_y
#gaze_z = right_eye_center[2] - left_eye_center[2] ? 

#TEGA UPORABLAM
new_df = pd.DataFrame({
    "left_eye_x": left_eye_x_norm,
    "left_eye_y": left_eye_y_norm,
    "right_eye_x": right_eye_x_norm,
    "right_eye_y": right_eye_y_norm,
    
    "nose_tip_x": nose_tip_x_norm,
    "nose_tip_y": nose_tip_y_norm,

    "mouth_x": mouth_x_norm,
    "mouth_y": mouth_y_norm,
    #"mouth_center": mouth_center,
    "head_pitch": pitch_proxy,
    "head_yaw": yaw_proxy,
    "head_roll": roll_proxy,
    # "head_pitch": df["head_pitch"],
    # "head_yaw": df["head_yaw"],
    # "head_roll": df["head_roll"],
    "label": df["label"]
}, index=df.index)

X = new_df.drop(columns="label")
y = new_df["label"]



X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


#binarni target column za attentiveness je label
target_column = 'label'

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

###########za 10 krizno validacijo###########
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    model,
    X_dev,
    y_dev,
    cv=cv,
    scoring=[
        "accuracy",
        "precision",
        "recall",
        "f1"
    ],
    return_train_score=True
)

print(y_dev.value_counts())
print(y_dev.value_counts(normalize=True))

print("###10FOLD")
print("Accuracy:", scores["test_accuracy"])
print("Mean accuracy:", scores["test_accuracy"].mean())
print("Std accuracy:", scores["test_accuracy"].std())

print("Mean precision:", scores["test_precision"].mean())
print("Mean recall:", scores["test_recall"].mean())
print("Mean F1:", scores["test_f1"].mean())
print("Std F1:", scores["test_f1"].std())
print("###10FOLD")


print("Training SVM...")

model.fit(X_dev, y_dev)

print("Training finished, saving model...")

joblib.dump(model, "attention_svm.joblib")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("RESULTS")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)

plt.title("SVM Attention Classification")
plt.show()