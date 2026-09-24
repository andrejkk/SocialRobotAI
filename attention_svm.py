from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_data():
    # nalozi daiseee ali student attentive dataset
    pass

model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        probability=True
    )
)

model.fit(X_train, y_train)