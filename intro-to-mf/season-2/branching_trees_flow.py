from metaflow import FlowSpec, step, Parameter

class ParallelTreesFlow(FlowSpec):

    test_size = Parameter("tst-sz", default=0.2)
    random_state = Parameter("seed", default=21)
    n_estimators = Parameter("n-est", default=10)
    min_samples_split = Parameter("min-samples", default=2)
    eval_metric = Parameter("eval-metric", default='mlogloss')

    @step
    def start(self):
        from sklearn_data_loader import get_iris
        self.X, self.y, self.X_train, self.y_train, self.X_test, self.y_test, _ = get_iris(
            self.test_size, self.random_state)
        self.next(self.train_rf, self.train_xgb)

    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          min_samples_split=self.min_samples_split, 
                                          random_state=self.random_state)
        self.clf.fit(self.X_train, self.y_train)
        self.next(self.score)

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
    def score(self, inputs):
        self.merge_artifacts(inputs, include=["X_test", "y_test"])
        self.accuracies = [
            train_step.clf.score(self.X_test, self.y_test)
            for train_step in inputs
        ]
        self.next(self.end)

    @step
    def end(self):
        self.model_names = ["Random Forest", "XGBoost"]
        for name, acc in zip(self.model_names, self.accuracies):
            print("{} Model Accuracy: {}%".format(
                name, round(100*acc, 3)))

if __name__ == "__main__":
    ParallelTreesFlow()
