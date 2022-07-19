
from metaflow import FlowSpec, step, Flow, current

class MyFlow(FlowSpec):

    @step
    def start(self):
        "Read the data"
        import pandas as pd
        self.df = pd.read_parquet('train.parquet')
        print(f'num of rows: {self.df.shape[0]}')
        self.next(self.baseline, self.train)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.metrics import accuracy_score, roc_auc_score
        baseline_predictions = [1] * self.df.shape[0]
        self.base_acc = accuracy_score(self.df.labels, baseline_predictions)
        self.base_rocauc = roc_auc_score(self.df.labels, baseline_predictions)
        self.next(self.join)

    @step
    def train(self):
        "Train the model"
        import tensorflow as tf
        from tensorflow.keras.utils import set_random_seed
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.feature_extraction.text import CountVectorizer
        from model import get_model
        set_random_seed(2022)
        
        self.cv = CountVectorizer(min_df=.005, max_df = .75, stop_words='english', strip_accents='ascii', )
        res = self.cv.fit_transform(self.df['review'])
        self.model = get_model(len(self.cv.vocabulary_))
        self.model.fit(x=res.toarray(), 
                       y=self.df['labels'],
                       batch_size=32, epochs=10, validation_split=.2)

        self.next(self.join)
        
    @step
    def join(self, inputs):
        "Compare the model results with the baseline."
        import tensorflow as tf
        from tensorflow.keras import layers, optimizers, regularizers
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.feature_extraction.text import CountVectorizer
        import pandas as pd
        
        
        self.model = inputs.train.model
        self.cv = inputs.train.cv
        self.train_df = inputs.train.df
        self.holdout_df = pd.read_parquet('holdout.parquet')
        
        self.predictions = self.model.predict(self.cv.transform(self.holdout_df['review']).toarray())
        labels = self.holdout_df['labels']
        
        self.model_acc = accuracy_score(labels, self.predictions > .5)
        self.model_rocauc = roc_auc_score(labels, self.predictions)
        
        print(f'Baseline Acccuracy: {inputs.baseline.base_acc:.2%}')
        print(f'Baseline AUC: {inputs.baseline.base_rocauc:.2}')
        print(f'Model Acccuracy: {self.model_acc:.2%}')
        print(f'Model AUC: {self.model_rocauc:.2}')
        self.beats_baseline = self.model_rocauc > inputs.baseline.base_rocauc
        print(f'Model beats baseline (T/F): {self.beats_baseline}')
        
        #smoke test to make sure model is doing the right thing on obvious examples.
        _tst_reviews = ["poor fit its baggy in places where it isn't supposed to be.",
                        "love it, very high quality and great value"]
        _tst_preds = self.model.predict(self.cv.transform(_tst_reviews).toarray())
        self.passed_smoke_test = _tst_preds[0][0] < .5 and _tst_preds[1][0] > .5
        print(f'Model passed smoke test (T/F): {self.passed_smoke_test}')
        
        self.next(self.retrain)

    @step
    def retrain(self):
        "If model beats the baseline and passes smoke tests, then retrain the model on all available data."
        if self.beats_baseline and self.passed_smoke_test:
            from sklearn.feature_extraction.text import CountVectorizer
            from tensorflow.keras.utils import set_random_seed
            import pandas as pd
            from model import get_model
            
            set_random_seed(2022)
            all_df = pd.concat([self.train_df, self.holdout_df])
            res = self.cv.transform(all_df['review'])
            self.final_model = get_model(len(self.cv.vocabulary_))

            self.final_model.fit(x=res.toarray(), 
                                 y=all_df['labels'],
                                 batch_size=32, epochs=10, validation_split=.1)
            run = Flow(current.flow_name)[current.run_id]
            run.add_tag('deployment_candidate')
        else:
            print('Model was not retrained on full data because of failed smoke test or performance below the baseline.')
        self.next(self.end)
        
    @step
    def end(self): ...

if __name__ == '__main__':
    MyFlow()
