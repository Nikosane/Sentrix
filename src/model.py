from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

def train_model(X_train, y_train, model_type="random_forest"):
    """
    Train a machine learning model on the training data.
    """
    if model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = OneClassSVM()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
