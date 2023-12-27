import numpy as np


class SVDClassifier:
    def __init__(self):
        self.U_n = {}

    def fit(self, X_train, Y_train):
        for label in np.unique(Y_train):
            U, _, _ = np.linalg.svd(X_train[:, Y_train == label], full_matrices=False)
            self.U_n[label] = U
        return self

    def predict(self, new_data):
        best_dist = +np.inf
        prediction = None

        for label in self.U_n:
            U = self.U_n[label]
            proj = U @ (U.T @ new_data)

            dist = np.linalg.norm(new_data - proj, 2)
            if dist < best_dist:
                best_dist = dist
                prediction = label

        return prediction