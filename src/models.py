import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

class OrdinalLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Cumulative link ordinal classifier using the Frank & Hall (2001) decomposition.
    Transforms an K-class ordinal problem (1 < 3 < 5) into (K - 1) binary classifiers
    predicting P(Y > r), preserving the natural ordering of biological stress tolerance.
    """
    def __init__(self, C=1.0, class_weight='balanced', random_state=42):
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.models_ = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.models_ = []
        # Train (K - 1) binary classifiers: P(Y > class_k)
        for i in range(len(self.classes_) - 1):
            threshold = self.classes_[i]
            y_binary = (y > threshold).astype(int)
            model = LogisticRegression(C=self.C, class_weight=self.class_weight,
                                       max_iter=1000, random_state=self.random_state)
            model.fit(X, y_binary)
            self.models_.append(model)
        return self

    def predict_proba(self, X):
        # Compute cumulative probabilities P(Y > r)
        p_greater = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        probs = np.zeros((X.shape[0], len(self.classes_)))
        # P(Y = Class 0) = 1 - P(Y > Class 0)
        probs[:, 0] = 1.0 - p_greater[:, 0]
        # P(Y = Class i) = P(Y > Class i-1) - P(Y > Class i)
        for i in range(1, len(self.classes_) - 1):
            probs[:, i] = np.clip(p_greater[:, i - 1] - p_greater[:, i], 0.0, 1.0)
        # P(Y = Class K-1) = P(Y > Class K-2)
        probs[:, -1] = p_greater[:, -1]
        
        # Normalize rows to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def get_model_suite(random_state=42):
    """
    Returns a dictionary of all evaluated classification models.
    """
    return {
        'Dummy (Baseline)': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state),
        'Support Vector Machine (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state),
        'Ordinal Logistic Regression': OrdinalLogisticRegression(C=1.0, class_weight='balanced', random_state=random_state)
    }
