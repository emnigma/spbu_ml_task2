import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class KNN(BaseEstimator):
    def __init__(self, k=3, p_minkowski=2):
        self.k = k
        self.p_minkowski = p_minkowski

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        # get predictions for every row in test data
        y_pred = [self._get_single_prediction(x_test_row) for x_test_row in X]

        return np.array(y_pred)

    def _get_single_prediction(self, x_test_row):
        # compute dist against test subj
        distances = [
            self._get_minkowski_distance(
                x_test_row, x_train_row, self.p_minkowski)
            for x_train_row in self.X_
        ]
        # get indices of k-nearest neighbors -> k-smallest distances
        k_idx = np.argsort(distances)[:self.k]
        # get corresponding y-labels of training data
        k_labels = [self.y_[idx] for idx in k_idx]
        # return most common label index
        return np.argmax(np.bincount(k_labels))

    def _get_minkowski_distance(self, x1, x2, p):
        # when p=1: manhattan dist
        # when p=2: euclidean dist

        dist_sum = np.abs(np.sum((x1 - x2)**p))
        return np.power(dist_sum, 1/p)
    
    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return accuracy_score(y_test, preds)


def main():
    # Testing
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    scores = []

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (idx_train, idx_valid) in enumerate(cv.split(X)):
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = KNN(k=3, p_minkowski=2)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_valid)

        score = accuracy(y_valid, predictions)
        scores.append(score)

    print(f"Mean Accuracy: {np.mean(scores)}")


if __name__ == "__main__":
    main()
    check_estimator(KNN())
