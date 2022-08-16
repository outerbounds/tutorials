from metaflow import FlowSpec, step

class ArtifactFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.create_artifact)

    @step
    def create_artifact(self):
        self.dataset = [[1,2,3], [4,5,6], [7,8,9]]
        self.metadata_description = "created"
        self.next(self.transform_artifact)

    @step
    def transform_artifact(self):
        self.dataset = [
            [value * 10 for value in row] 
            for row in self.dataset
        ]
        self.metadata_description = "transformed"
        self.next(self.end)

    @step
    def end(self):
        print("Artifact is in state `{}` with values {}".format(
            self.metadata_description, self.dataset))

if __name__ == "__main__":
    ArtifactFlow()
