from metaflow import FlowSpec, step

class Hello(FlowSpec):
    
    @step
    def start(self):
        print("Hello World")
        self.next(self.end)
        
    @step
    def end(self):
        print("Flow is done")
    
    
if __name__ == "__main__":
    Hello()