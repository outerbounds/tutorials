
from metaflow import FlowSpec, step, Flow, current

class NLPPredictionFlow(FlowSpec):
    
    def get_latest_successful_run(self, flow_nm, tag):
        """Gets the latest successful run 
            for a flow with a specific tag."""
        for r in Flow(flow_nm).runs(tag):
            if r.successful: return r
        
    @step
    def start(self):
        """Get the latest deployment candidate 
            that is from a successfull run"""
        self.deploy_run = self.get_latest_successful_run(
            'NLPFlow', 'deployment_candidate')
        self.next(self.end)
    
    @step
    def end(self):
        "Make predictions"
        from model import NbowModel
        import pandas as pd
        import pyarrow as pa
        new_reviews = pd.read_parquet(
            'predict.parquet')['review']
        
        # Make predictions
        model = NbowModel.from_dict(
            self.deploy_run.data.model_dict)
        predictions = model.predict(new_reviews)
        msg = 'Writing predictions to parquet: {} rows'
        print(msg.format(predictions.shape[0]))
        pa_tbl = pa.table({"data": predictions.squeeze()})
        pa.parquet.write_table(
            pa_tbl, "sentiment_predictions.parquet")
        
if __name__ == '__main__':
    NLPPredictionFlow()
