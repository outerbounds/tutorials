from metaflow import FlowSpec, step

class DebuggableFlow(FlowSpec):
    
    @step
    def start(self):
        self.next(self.time_consuming_step)
        
    @step
    def time_consuming_step(self):
        import time
        time.sleep(12)
        self.next(self.error_prone_step)
        
    @step
    def error_prone_step(self):
        #highlight-next-line
        print("Squashed bug")
        # raise Exception()
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow is done!")

if __name__ == "__main__":
    DebuggableFlow()
