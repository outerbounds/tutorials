from metaflow import FlowSpec, step, Parameter

class RandomForestFlow(FlowSpec):

    test_size = Parameter("tst-sz", default=0.2)
    random_state = Parameter("seed", default=11)
    n_estimators = Parameter("n-est", default=10)
    min_samples_split = Parameter("min-samples", 
                                  default=2)

    @step
    def start(self):
        from sklearn_data_loader import get_iris
        (self.X, self.y, self.X_train, 
        self.y_train, self.X_test, self.y_test, 
        _) = get_iris(self.test_size, self.random_state)
        self.next(self.train_rf)

    @step
    def train_rf(self):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split, 
            random_state=self.random_state)
        self.clf.fit(self.X_train, self.y_train)
        self.next(self.score)

    @step
    def score(self):
        self.accuracy = self.clf.score(self.X_test, 
                                       self.y_test)
        self.next(self.end)

    @step
    def end(self):
        print("Random Forest Accuracy: {}%".format(
            round(100*self.accuracy, 3)))

if __name__ == "__main__":
    RandomForestFlow()
