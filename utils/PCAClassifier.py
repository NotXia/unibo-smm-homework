import numpy as np


def _centroid(data):
    return np.expand_dims( np.mean(data, axis=1), 1 )

class _PCA:
    def __init__(self):
        self.data_centroid = None
        self.proj_matrix = None

    def fit(self, data, k):
        self.data_centroid = _centroid(data)
        centered_data = data - self.data_centroid

        U, _, _ = np.linalg.svd(centered_data, full_matrices=False)
        self.proj_matrix = U[:, :k].T

    def transform(self, data):
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
            
        data_centered = data - self.data_centroid
        return self.proj_matrix @ data_centered
    
    def fit_transform(self, data, k):
        self.fit(data, k)
        return self.transform(data)


class PCAClassifier:
    def __init__(self):
        self.pca = None
        self.Z_k_train = None
        self.possible_labels = None
        self.labels_centroid = None

    def fit(self, X_train, Y_train, k=15):
        self.pca = _PCA()
        self.Z_k_train = self.pca.fit_transform(X_train, k)
        self.possible_labels = np.unique(Y_train)
        self.labels_centroid = { label: _centroid(self.Z_k_train[:, Y_train == label]) for label in self.possible_labels }
        return self

    def predict(self, new_data):
        Z_k_data = self.pca.transform(new_data)
        best_distance = +np.inf
        best_label = None

        for label in self.possible_labels:
            assert self.labels_centroid[label].shape == Z_k_data.shape
            distance = np.linalg.norm(self.labels_centroid[label] - Z_k_data, 2)
            if distance < best_distance:
                best_distance = distance
                best_label = label

        return best_label