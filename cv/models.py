
from tensorflow import keras

class ModelOperations:

    recall = 0.96
    precision = 0.96

    def make_baseline(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            self.num_pixels, input_dim=self.num_pixels,
            kernel_initializer=self.kernel_initializer,
            activation='relu'
        ))
        model.add(keras.layers.Dense(
            self.num_classes,
            kernel_initializer=self.kernel_initializer,
            activation='softmax'
        ))
        model.compile(
            loss=self.loss, 
            optimizer=self.optimizer, 
            metrics=self._keras_metrics()
        )
        return model

    def make_cnn(
        self,
        hidden_conv_layer_sizes = None
    ):
        model = keras.Sequential()
        if hidden_conv_layer_sizes is None:
            hidden_conv_layer_sizes = self.hidden_conv_layer_sizes
        _layers = [keras.Input(shape=self.input_shape)]
        for conv_layer_size in hidden_conv_layer_sizes:
            _layers.append(keras.layers.Conv2D(
                conv_layer_size, 
                kernel_size=self.kernel_size, 
                activation="relu"
            ))
            _layers.append(keras.layers.MaxPooling2D(
                pool_size=self.pool_size
            ))
        _layers.extend([
            keras.layers.Flatten(),
            keras.layers.Dropout(self.p_dropout),
            keras.layers.Dense(self.num_classes, activation="softmax")
        ])
        model = keras.Sequential(_layers)
        model.compile(
            loss=self.loss, 
            optimizer=self.optimizer, 
            metrics=self._keras_metrics()
        )
        return model

    def fit_and_score(self, x_train, x_test):
        history = self.model.fit(
            x_train, self.y_train,
            validation_data = (x_test, self.y_test),
            epochs = self.epochs,
            batch_size = self.batch_size,
            verbose = self.verbose
        )
        scores = self.model.evaluate(x_test, self.y_test, verbose = 0)
        return history, scores

    def _keras_metrics(self):
        keras_metrics = []
        for _m in self.metrics:
            if _m == 'precision at recall':
                keras_metrics.append(
                    keras.metrics.PrecisionAtRecall(recall=self.recall)
                )
            elif _m == 'recall at precision':
                keras.metrics.append(
                    keras.metrics.RecallAtPrecision(precision=self.precision)
                )
            else:
                keras_metrics.append(_m)
        return keras_metrics
