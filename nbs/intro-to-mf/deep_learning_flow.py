from metaflow import FlowSpec, step, Parameter
from tensorflow import keras

def build_model(hidden_layer_dim, meta):
    # meta is a scikeras argument that will be
    # handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    # build neural net model 
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, 
                                 input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_layer_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model

class DeepLearningFlow(FlowSpec):
    
    test_size = Parameter("test_size", default=0.2)
    random_state = Parameter("random_state", default=42)
    hidden_layer_dim = Parameter("hidden_layer_dim", default=100)
    epochs = Parameter("epochs", default=200)
    loss_fn = Parameter("loss_fn", default='categorical_crossentropy')
    
    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        iris = datasets.load_iris()
        self.X = iris['data']
        self.y = iris['target']
        data = train_test_split(self.X, self.y, 
                                test_size=self.test_size, 
                                random_state=self.random_state)
        self.X_train = data[0]
        self.X_test = data[1]
        self.y_train = data[2]
        self.y_test = data[3]        
        self.next(self.scale_features)
        
    @step
    def scale_features(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.next(self.train)
        
    @step
    def train(self):
        from scikeras.wrappers import KerasClassifier
        self.clf = KerasClassifier(build_model, 
                                   loss=self.loss_fn,
                                   hidden_layer_dim=self.hidden_layer_dim,
                                   epochs=self.epochs,
                                   verbose=0)
        self.clf.fit(self.X_train, self.y_train)
        self.next(self.score)

    @step
    def score(self):
        self.accuracy = self.clf.score(self.X_test, self.y_test)
        self.next(self.end)
    
    @step
    def end(self):
        print("Neural Net Model Accuracy: {}%".format(round(100*self.accuracy, 3)))
        
if __name__ == "__main__":
    DeepLearningFlow()
