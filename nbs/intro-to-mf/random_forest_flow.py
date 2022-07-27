from metaflow import FlowSpec, step, Parameter

class RandomForestFlow(FlowSpec):
    
    test_size = Parameter("test_size", default=0.2)
    random_state = Parameter("random_state", default=4)
    n_estimators = Parameter("n_estimators", default=10)
    min_samples_split = Parameter("min_samples_split", default=2)
    
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
        self.next(self.train)
        
    @step
    def train(self):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                          min_samples_split=self.min_samples_split, 
                                          random_state=self.random_state)
        self.clf.fit(self.X_train, self.y_train)
        self.next(self.score)

    @step
    def score(self):
        self.accuracy = self.clf.score(self.X_test, self.y_test)
        self.next(self.end)
    
    @step
    def end(self):
        print("Random Forest Model Accuracy: {}%".format(round(100*self.accuracy, 3)))
        
if __name__ == "__main__":
    RandomForestFlow()