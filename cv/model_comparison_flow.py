from metaflow import FlowSpec, step, Flow, current, card
from metaflow.cards import Image, Table
from tensorflow import keras
from models import ModelOperations

class ModelComparisonFlow(FlowSpec, ModelOperations):

    best_model_location = ("latest_image_classifier")
    num_pixels = 28 * 28
    kernel_initializer = 'normal'
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy', 'precision at recall']
    hidden_conv_layer_sizes = [32, 64]
    input_shape = (28, 28, 1)
    kernel_size = (3, 3)
    pool_size = (2, 2)
    p_dropout = 0.5
    epochs = 6
    batch_size = 64
    verbose = 2 

    @step
    def start(self):
        import numpy as np
        self.num_classes = 10
        ((x_train, y_train), 
         (x_test, y_test)) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        self.x_train = np.expand_dims(x_train, -1)
        self.x_test = np.expand_dims(x_test, -1)
        self.y_train = keras.utils.to_categorical(
            y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(
            y_test, self.num_classes)
        self.num_classes = self.y_test.shape[1]
        self.next(self.baseline, self.cnn)

    @step
    def baseline(self):
        from neural_net_utils import plot_learning_curves
        self.model = self.make_baseline()
        _x_train = self.x_train.reshape(
            self.x_train.shape[0], self.num_pixels
        ).astype('float32')
        _x_test = self.x_test.reshape(
            self.x_test.shape[0], self.num_pixels
        ).astype('float32')
        self.history, self.scores = self.fit_and_score(
            _x_train, _x_test)
        self._name = "Baseline FFN"
        self.plots = [
            Image.from_matplotlib(p) for p in
            plot_learning_curves(self.history, self._name)
        ]
        self.next(self.gather_scores)
    
    @step
    def cnn(self):
        from neural_net_utils import plot_learning_curves
        self.model = self.make_cnn()
        #highlight-start
        self.history, self.scores = self.fit_and_score(
            self.x_train, self.x_test)
        #highlight-end
        self._name = "CNN"
        self.plots = [
            Image.from_matplotlib(p) for p in
            plot_learning_curves(self.history, self._name)
        ]
        self.next(self.gather_scores)

    
    @card
    @step
    def gather_scores(self, models):
        import pandas as pd
        results = {
            'model': [], 'test loss': [],
            **{metric: [] for metric in self.metrics}
        }
        max_seen_acc = 0
        rows = []
        for model in models:
            #highlight-start
            results['model'].append(model._name)
            results['test loss'].append(model.scores[0])
            for i, metric in enumerate(self.metrics):
                results[metric].append(model.scores[i+1])
            #highlight-end
            rows.append(model.plots)
            if model.scores[1] > max_seen_acc:
                self.best_model = model.model
                max_seen_acc = model.scores[1]
        #highlight-next-line
        current.card.append(Table(rows))
        self.results = pd.DataFrame(results)
        self.next(self.end)

    @step
    def end(self):
        self.best_model.save(self.best_model_location)

if __name__ == '__main__':
    ModelComparisonFlow()
