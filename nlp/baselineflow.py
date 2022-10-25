
from metaflow import FlowSpec, step, Flow, current

class BaselineNLPFlow(FlowSpec):

    @step
    def start(self):
        "Read the data"
        import pandas as pd
        self.df = pd.read_parquet('train.parquet')
        self.valdf = pd.read_parquet('valid.parquet')
        print(f'num of rows: {self.df.shape[0]}')
        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.metrics import accuracy_score, roc_auc_score
        baseline_predictions = [1] * self.valdf.shape[0]
        self.base_acc = accuracy_score(
            self.valdf.labels, baseline_predictions)
        self.base_rocauc = roc_auc_score(
            self.valdf.labels, baseline_predictions)
        self.next(self.end)

    @step
    def end(self):
        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

if __name__ == '__main__':
    BaselineNLPFlow()
