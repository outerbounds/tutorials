from metaflow import FlowSpec, step, Parameter

class ParameterizedFlow(FlowSpec):
    
    #highlight-next-line
    learning_rate = Parameter('lr', default=.01)
    
    @step
    def start(self):
        self.next(self.end)
    
    @step
    def end(self):
        print("Learning rate value is {}".format(self.learning_rate))

if __name__ == "__main__":
    ParameterizedFlow()
