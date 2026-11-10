from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, GridSearchCV, cross_validate

from scipy.sparse.csgraph import connected_components


from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans

import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score

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

####claude agent NI bil uporabljen

def load_data():
    try:
        df = pd.read_csv('trainv3/attention_detection_dataset_v3.csv',
                     delimiter=',', encoding='ISO-8859-1')
    except FileNotFoundError:
        print("napaka pri branju datoteke, preveri pot do datoteke")
        return None

    df.drop(['no_of_face','hand_count','left_hand_x','left_hand_y','right_hand_x','right_hand_y','hand_obj_interaction','head_pose','phone_present', 'phone_loc_x', 'phone_loc_y', 'phone_conf', 'gaze_on_screen', 'gaze_direction','gazePoint_x','gazePoint_y', 'pupil_left_x','pupil_left_y','pupil_right_x','pupil_right_y']
    ,axis=1, inplace=True)

    return df

df = load_data()

#TODO: a iz teh raj nov df zgradim al updateam izvornega??
df = df[df['face_present'] == 1].copy()


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


gaze_x = eye_mid_x - eye_mid_x
gaze_y = eye_mid_y - eye_mid_y
#gaze_z = right_eye_center[2] - left_eye_center[2] ? 

##############################
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

# groups = pd.Series(
#     pd.factorize(
#         pd.MultiIndex.from_frame(X)
#     )[0],
#     index=X.index
# )
# Zc = StandardScaler().fit_transform(df[['face_x', 'face_y', 'face_w', 'face_h']])
# groups = pd.Series(KMeans(20, n_init=5, random_state=0).fit_predict(Zc), index=X.index)


#X_scaled = StandardScaler().fit_transform(df[['face_x', 'face_y', 'face_w', 'face_h', 'left_eye_x', 'right_eye_x', 'mouth_x', 'mouth_y', 'nose_tip_x', 'nose_tip_y']])
X_scaled = StandardScaler().fit_transform(X)


nn = NearestNeighbors(
    metric="euclidean",
    n_jobs=-1
)

nn.fit(X_scaled)
eps_values = [
    0.01, 0.02, 0.05, 0.1,
    0.15, 0.2, 0.3, 0.4, 0.5
]

for eps in eps_values:

    graph = nn.radius_neighbors_graph(
        X_scaled,
        radius=eps,
        mode="connectivity"
    )

    graph = graph.maximum(graph.T)

    n_groups, group_ids = connected_components(
        graph,
        directed=False
    )

    sizes = pd.Series(group_ids).value_counts()

    print(
        f"EPS: {eps:.2f} | "
        f"groups: {n_groups} | "
        f"max: {sizes.max()} | "
        f"min: {sizes.min()} | "
        f"Median: {sizes.median():.1f} | "
        f"najvecja: {sizes.max()/len(X):.2%}"
    )

# nn_d = NearestNeighbors(
#     n_neighbors=2,
#     metric="euclidean"
# )
# nn_d.fit(X_scaled)
# distances, ndices = nn_d.kneighbors()

# nearest_distances = distances[:, 0]

# print(
#     pd.Series(nearest_distances).describe(
#         offset =[
#             0.10, 0.25, 0.50,
#             0.75, 0.90, 0.95, 0.99
#         ]
#     )
# )

#eps_thresh = 0.10
eps_thresh = 0.15

# povezi najblizje tko z eps
graph = nn.radius_neighbors_graph(
    X_scaled,
    radius=eps_thresh,
    mode="connectivity"
)

graph = graph.maximum(graph.T)

n_groups, group_ids = connected_components(
    graph,
    directed=False
)

#X ne x scaled!!
groups = pd.Series(group_ids, index=X.index, name="similarity_group")

# print("Number of groups:", n_groups)
# print("Largest group:", groups.value_counts().max())
# print("Smallest group:", groups.value_counts().min())
# Inspect grouping
sizes = groups.value_counts()

print("\nGROUPING RESULTS")
print("Observations:", len(X))
print("Groups:", n_groups)
print("Largest group:", sizes.max())
print("Median group size:", sizes.median())
print("Singleton groups:", (sizes == 1).sum())
#tega uporablam
skgf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
dev_idx, test_idx = next(skgf.split(X, y, groups))
#dev_idx, test_idx = next(skgf.split(X, y, groups=groups)) al tko?

# assert set(groups.iloc[dev_idx]]).isdisjoint(set(groups[test_idx]))
assert set(groups.iloc[dev_idx]).isdisjoint(set(groups.iloc[test_idx]))
X_dev, X_test = X.iloc[dev_idx], X.iloc[test_idx]
y_dev, y_test = y.iloc[dev_idx], y.iloc[test_idx]
g_dev, g_test = groups.iloc[dev_idx], groups.iloc[test_idx]

# preveri prekrivanje ce je ok
assert set(g_dev).isdisjoint(set(g_test))
print("\nTRAIN / TEST SPLIT")
print("Development samples:", len(X_dev))
print("Test samples:", len(X_test))
print("\nDevelopment labels:")
print(y_dev.value_counts())
print("\nTest labels:")
print(y_test.value_counts())
print("Overlapping groups:", len(set(g_dev) & set(g_test)))

#bols 10 za finetuning
cv5  = StratifiedGroupKFold(n_splits=5,  shuffle=True, random_state=42) 
cv10 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

# grid = GridSearchCV(
#     make_pipeline(StandardScaler(), SVC()),
#     {"svc__kernel":["linear", "rbf", "sigmoid"], "svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.03, "scale", 0.3, 1], "svc__class_weight": [None, "balanced"]},
#     cv=cv5, scoring="balanced_accuracy", n_jobs=-1, refit=True,
# )
pipeline = make_pipeline(
    StandardScaler(),
    SVC(probability=True)
)

param_grid = {
    "svc__kernel": ["linear", "rbf"],
    "svc__C": [0.1, 1, 10, 100],
    #"svc__C": [0.01, 0.1, 1, 10, 100, 1000],
    "svc__gamma": [0.03, "scale", 0.3, 0.5, 1],
    #"svc__gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 0.5, 1],
    "svc__class_weight": [None, "balanced"]
}

finetune_scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}

cv_splits = list(
    cv10.split(X_dev, y_dev, groups=g_dev)
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_splits,
    scoring=finetune_scoring,
    n_jobs=-1,
    refit="balanced_accuracy"
)



grid.fit(X_dev, y_dev)
best = grid.best_params_
#print("\nbest on dev:", best, "| dev CV balanced accuracy: %.3f" % grid.best_score_)
print("\nbest on dev:", best)
print("Best parameters:", grid.best_params_)
print("Best balanced accuracy:", grid.best_score_)


#model = make_pipeline(StandardScaler(), SVC(
    #kernel=best["svc__kernel"], C=best["svc__C"], gamma=best["svc__gamma"], class_weight=best["svc__class_weight"]))
model = grid.best_estimator_

target_column = 'label'

best_idx = grid.best_index_

print("\n########### SVM 10-FOLD CV ###########")

for metric in scoring:
    mean = grid.cv_results_[f"mean_test_{metric}"][best_idx]
    std = grid.cv_results_[f"std_test_{metric}"][best_idx]

    print(f"{metric}: {mean:.4f} (+/- {std:.4f})")

# ==========================================
# FINAL SVM TEST
# ==========================================

y_pred = model.predict(X_test)

print("\n_______SVM TEST RESULTS_________")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced accuracy:",
      balanced_accuracy_score(y_test, y_pred))

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Attentive", "Inattentive"],
    zero_division=0
))

print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("MAJORITY")

majority_model = DummyClassifier(
    strategy="most_frequent"
)

class_counts = y_dev.value_counts().sort_index()

print("\n_______CLASS DISTRIBUTION _______")
print(class_counts)

majority_class = class_counts.idxmax()

class_names = {
    0: "Attentive",
    1: "Inattentive"
}

print(
    f"Majority class: {majority_class} "
    f"({class_names[majority_class]})"
)
for i, (train_idx, val_idx) in enumerate(cv_splits):

    fold_majority = (
        y_dev.iloc[train_idx].value_counts().idxmax()
    )

    print(f"Fold {i+1}: {class_names[fold_majority]}")


majority_model.fit(X_dev, y_dev)

y_pred_majority = majority_model.predict(X_test)

print("\n__________MAJORITY BASELINE TEST RESULTS__________")
print("Accuracy:",
      accuracy_score(y_test, y_pred_majority))

print(classification_report(
    y_test,
    y_pred_majority,
    labels=[0, 1],
    target_names=["Attentive", "Inattentive"],
    zero_division=0
))

print(confusion_matrix(
    y_test,
    y_pred_majority,
    labels=[0, 1]
))

joblib.dump(model, "attention_svm.joblib")
#majorityvote = DummyClassifier(strategy="most_frequent")
#tuki nekje morm dt macro za scoring? TODO

# cv_splits = list(
#     cv10.split(X_dev, y_dev, groups=g_dev)
# )

base_scores = cross_validate(
    majority_model,
    X_dev,
    y_dev,
    cv=cv_splits,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)


# scores = cross_validate(model, X_dev, y_dev, cv=cv10, groups=g_dev, scoring=["accuracy","precision_macro","recall_macro","f1_macro"],return_train_score=True)
# base_scores = cross_validate(majority_model, X_dev, y_dev, cv=cv10, groups=g_dev, scoring=["accuracy","precision_macro","recall_macro","f1_macro"],return_train_score=True)
#scores = cross_validate(model, X_dev, y_dev, cv=cv_splits, scoring=scoring,return_train_score=True)
#base_scores = cross_validate(majority_model, X_dev, y_dev, cv=cv_splits, groups=g_dev, scoring=scoring,return_train_score=True)

# print("=============SVM FINAL 10FOLD VALIDATE==========")
# print("Mean accuracy:", scores["test_accuracy"].mean())
# print("Std accuracy:", scores["test_accuracy"].std())
# print("Mean precision:", scores["test_precision_macro"].mean())
# print("Mean recall:", scores["test_recall_macro"].mean())
# print("Mean F1:", scores["test_f1_macro"].mean())
# #print(scores["test_accuracy"].mean())
# print("=============SVM FINAL 10FOLD VALIDATE==========")

print("=============MAJORITY VOTE==========")
print("Mean accuracy:", base_scores["test_accuracy"].mean())
print("Std accuracy:", base_scores["test_accuracy"].std())
print("Mean precision:", base_scores["test_precision_macro"].mean())
print("Mean recall:", base_scores["test_recall_macro"].mean())
print("Mean F1:", base_scores["test_f1_macro"].mean())
#print(base_scores)

print("\n========== MAJORITY BASELINE 10-FOLD CV ==========")

for metric in scoring:

    mean = base_scores[f"test_{metric}"].mean()
    std = base_scores[f"test_{metric}"].std()

    print(f"{metric}: {mean:.4f} (+/- {std:.4f})")


#print("=============MAJORITY VOTE==========")

# majorityvote.fit(X_dev, y_dev)
# y_pred_base = majorityvote.predict(X_test)

# print("Baseline accuracy:", accuracy_score(y_test, y_pred_base))
# print(classification_report(y_test, y_pred_base, zero_division=0))
# print(confusion_matrix(y_test, y_pred_base))

# print("###10FOLD")
# print("Accuracy:", scores["test_accuracy"])
# print("Mean accuracy:", scores["test_accuracy"].mean())
# print("Std accuracy:", scores["test_accuracy"].std())

# print("Mean precision:", scores["test_precision"].mean())
# print("Mean recall:", scores["test_recall"].mean())
# print("Mean F1:", scores["test_f1"].mean())
# print("Std F1:", scores["test_f1"].std())
# print("###10FOLD")


# model.fit(X_dev, y_dev)

# print("Training finished, saving model...")

# joblib.dump(model, "attention_svm.joblib")


# y_pred = model.predict(X_test)


# accuracy = accuracy_score(y_test, y_pred)

# print("RESULTS")

# print(f"Accuracy: {accuracy:.4f}")

# print("\nClassification report:")
# print(classification_report(y_test, y_pred))

# print("\nConfusion matrix:")
# print(confusion_matrix(y_test, y_pred))


# ConfusionMatrixDisplay.from_predictions(
#     y_test,
#     y_pred
# )

# plt.title("SVM Attention Classification")
# plt.show()