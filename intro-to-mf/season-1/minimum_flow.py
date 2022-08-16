from metaflow import FlowSpec, step

class MinimumFlow(FlowSpec):
    
    @step
    def start(self):
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow is done!")

if __name__ == "__main__":
    MinimumFlow()
