#meta:tag=hide
from tensorflow import keras
import matplotlib.pyplot as plt

def build_model(hidden_layer_dim, meta):
    # meta is a scikeras argument that will be
    # handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_layer_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model

def get_feature_comparison_figure(X_train, X_train_scaled, X_test, X_test_scaled, feature_names, n_bins = 10):
    n_features = X_train.shape[1]
    assert n_features == X_test.shape[1], "Train and test feature dimensions are not the same!"
    feature_datasets = [X_train, X_train_scaled, X_test, X_test_scaled]  
    dataset_names = ["X_train", "X_train_scaled", "X_test", "X_test_scaled"]
    fig, axs = plt.subplots(len(feature_datasets), n_features, figsize=(16,16))
    for i,data in enumerate(feature_datasets):
        for j in range(n_features):
            axs[i,j].hist(data[:, j], bins=n_bins)
            axs[i,j].set_title("{} - {}".format(dataset_names[i], feature_names[j]))
    return fig
