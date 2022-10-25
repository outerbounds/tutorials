from metaflow import FlowSpec, step, Flow, current, card
from metaflow.cards import Image, Table
from tensorflow import keras
from models import ModelOperations

class TuningFlow(FlowSpec, ModelOperations):

    best_model_location = ("best_tuned_model")
    num_pixels = 28 * 28
    kernel_initializer = 'normal'
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = [
        'accuracy',
        'precision at recall'
    ]
    input_shape = (28, 28, 1)
    kernel_size = (3, 3)
    pool_size = (2, 2)
    p_dropout = 0.5
    epochs = 5
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
        #highlight-start
        self.param_config = [
            {"hidden_conv_layer_sizes": [16, 32]},
            {"hidden_conv_layer_sizes": [16, 64]},
            {"hidden_conv_layer_sizes": [32, 64]},
            {"hidden_conv_layer_sizes": [32, 128]},
            {"hidden_conv_layer_sizes": [64, 128]}
        ]
        self.next(self.train, foreach='param_config')
        #highlight-end

    @step
    def train(self):
        from neural_net_utils import plot_learning_curves
        #highlight-start
        self.model = self.make_cnn(
            self.input['hidden_conv_layer_sizes'])
        #highlight-end
        self.history, self.scores = self.fit_and_score(
            self.x_train, self.x_test)
        self._name = 'CNN'
        self.plots = [
            Image.from_matplotlib(p) for p in
            plot_learning_curves(
                self.history,
                'Hidden Layers - ' + ', '.join([
                    str(i) for i in
                    self.input['hidden_conv_layer_sizes']
                ])
            )
        ]
        self.next(self.gather_scores)

    @card
    @step
    def gather_scores(self, models):
        import pandas as pd
        self.max_class = models[0].y_train
        results = {
            'hidden conv layer sizes': [],
            'model': [], 
            'test loss': [],
            **{metric: [] for metric in self.metrics}
        }
        max_seen_acc = 0
        rows = []
        for model in models:
            results['model'].append(model._name)
            results['test loss'].append(model.scores[0])
            for i, metric in enumerate(self.metrics):
                results[metric].append(model.scores[i+1])
            results['hidden conv layer sizes'].append(
                ','.join([
                    str(i) for i in model.input[
                        'hidden_conv_layer_sizes'
                    ]
                ])
            )
            # A simple rule for determining the best model.
            # In production flows you need to think carefully
                # about how this kind of rule maps to your objectives.
            if model.scores[1] > max_seen_acc:
                self.best_model = model.model
                max_seen_acc = model.scores[1]
            rows.append(model.plots)

        #highlight-next-line
        current.card.append(Table(rows))
        self.results = pd.DataFrame(results)
        self.next(self.end)

    @step
    def end(self):
        self.best_model.save(self.best_model_location)

if __name__ == '__main__':
    TuningFlow()
