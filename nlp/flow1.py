
from metaflow import FlowSpec, step, Flow, current

class NLPFlow(FlowSpec):
        
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
        from model import Nbow_Model
        model = Nbow_Model(vocab_sz=750)
        model.fit(X=self.df['review'], y=self.df['labels'])
        self.model_dict = model.model_dict #save model
        self.next(self.join)
        
    @step
    def join(self, inputs):
        "Compare the model results with the baseline."
        import pandas as pd
        from model import NbowModel
        self.model_dict = inputs.train.model_dict
        self.train_df = inputs.train.df
        self.holdout_df = pd.read_parquet('holdout.parquet')
        model = NbowModel.from_dict(self.model_dict)
        
        self.model_acc = model.eval_acc(X=self.holdout_df['review'], labels=self.holdout_df['labels'])
        self.model_rocauc = model.eval_rocauc(X=self.holdout_df['review'], labels=self.holdout_df['labels'])
        
        print(f'Baseline Acccuracy: {inputs.baseline.base_acc:.2%}')
        print(f'Baseline AUC: {inputs.baseline.base_rocauc:.2}')
        print(f'Model Acccuracy: {self.model_acc:.2%}')
        print(f'Model AUC: {self.model_rocauc:.2}')
        self.next(self.end)
        
    @step
    def end(self):
        "Tags model as a deployment candidate if it beats the baseline and passes smoke tests."
        self.beats_baseline = self.model_rocauc > inputs.baseline.base_rocauc
        print(f'Model beats baseline (T/F): {self.beats_baseline}')
        #smoke test to make sure model is doing the right thing on obvious examples.
        _tst_reviews = ["poor fit its baggy in places where it isn't supposed to be.",
                        "love it, very high quality and great value"]
        _tst_preds = model.predict(_tst_reviews)
        self.passed_smoke_test = _tst_preds[0][0] < .5 and _tst_preds[1][0] > .5
        print(f'Model passed smoke test (T/F): {self.passed_smoke_test}')
        
        if self.beats_baseline and self.passed_smoke_test:
            run = Flow(current.flow_name)[current.run_id]
            run.add_tag('deployment_candidate')
        

if __name__ == '__main__':
    NLPFlow()
