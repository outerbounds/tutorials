from metaflow import FlowSpec, step, Parameter

class GradientBoostedTreesFlow(FlowSpec):

    test_size = Parameter("tst-sz", default=0.2)
    random_state = Parameter("seed", default=12)
    n_estimators = Parameter("n-est", default=10)
    min_samples_split = Parameter("min-samples", default=2)
    eval_metric = Parameter("eval-metric", default='mlogloss')

    @step
    def start(self):
        from sklearn_data_loader import get_iris
        self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test, _ = get_iris(
            self.test_size, self.random_state)
        self.next(self.train_xgb)

    @step
    def train_xgb(self):
        from xgboost import XGBClassifier
        self.clf = XGBClassifier(n_estimators=self.n_estimators,
                                 random_state=self.random_state,
                                 eval_metric=self.eval_metric,
                                 use_label_encoder=False)
        self.clf.fit(self.X_train, self.y_train)
        self.next(self.score)

    @step
    def score(self):
        self.accuracy = self.clf.score(self.X_test, self.y_test)
        self.next(self.end)

    @step
    def end(self):
        print("Gradient Boosted Trees Model Accuracy: {}%".format(round(100*self.accuracy, 3)))

if __name__ == "__main__":
    GradientBoostedTreesFlow()
