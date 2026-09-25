from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split

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
    df.drop(['face_present','no_of_face','hand_count','left_hand_x','left_hand_y','right_hand_x','right_hand_y','hand_obj_interaction','head_pose','phone_present', 'phone_loc_x', 'phone_loc_y', 'phone_conf', 'gaze_on_screen', 'gaze_direction','gazePoint_x','gazePoint_y','pupil_left_x','pupil_left_y','pupil_right_x','pupil_right_y']
    ,axis=1, inplace=True)

    print(df.head())    
    print(df[[
        "head_pitch",
        "head_yaw",
        "head_roll"
    ]].describe())

    return df

df = load_data()


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

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