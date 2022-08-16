from metaflow import FlowSpec, step, card


class DecoratorFlow(FlowSpec):
    
    @step
    def start(self):
        self.next(self.my_decorated_func)
        
    @card
    @step
    def my_decorated_func(self):
        self.data = [1, 2, 3]
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow is done!")

if __name__ == "__main__":
    DecoratorFlow()
