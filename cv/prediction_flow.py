from metaflow import FlowSpec, step, Flow, Parameter, current, card
import numpy as np

class SinglePredictionFlow(FlowSpec):

    upstream_flow = Parameter('flow', default = 'TuningFlow')
    image_location = Parameter('im', default = './mnist_random_img.npy')

    @step
    def start(self):
        from tensorflow import keras
        softmax = keras.layers.Softmax()
        run = list(Flow(self.upstream_flow).runs('production'))[-1]
        self.model = run.data.best_model 
        with open(self.image_location, 'rb') as f:
            self.image = np.load(f)
        self.logits = self.model.predict(x = np.array([self.image]))
        self.probs = softmax(self.logits).numpy()
        if np.isclose(1, np.sum(self.probs)):
            self.pred = self.probs.argmax()
        else:
            self.fallback_model = "Random Guess"
            self.pred = np.random.randint(low=0, high=9)
            print("{}/{} probabilities not adding to 1".format(
                self.__class__.__name__, current.run_id))
            print("Returning random fall back prediction")
        self.next(self.end)

    @card
    @step
    def end(self):
        import matplotlib.pyplot as plt
        from metaflow.cards import Table, Markdown, Image
        self.im_fig, self.im_ax = plt.subplots()
        self.im_ax.imshow(self.image, cmap='gray')
        im1 = Image.from_matplotlib(self.im_fig)
        md = Markdown("# Prediction: {}".format(self.pred))
        self.dist_fig, self.dist_ax = plt.subplots()
        self.dist_ax.barh(y=np.arange(
            self.probs[0].shape[0]), width=self.probs[0])
        self.dist_ax.set_yticks(
            np.arange(self.probs[0].shape[0]), 
            labels=np.arange(self.probs[0].shape[0])
        )
        self.dist_ax.set_ylabel('Probability', fontsize=18)
        im2 = Image.from_matplotlib(self.dist_fig)
        current.card.append(Table([[im1, md, im2]]))

if __name__ == '__main__': 
    SinglePredictionFlow()
