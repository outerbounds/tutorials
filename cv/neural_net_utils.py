import matplotlib.pyplot as plt
    
def plot_learning_curves(history, model_name):
    fig1, ax = plt.subplots(1,1)
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Accuracy Curve of {}'.format(model_name))
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    fig1.legend(['train', 'test'], loc='upper left')
    fig2, ax = plt.subplots(1,1)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Loss Curve of {}'.format(model_name))
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    fig2.legend(['train', 'test'], loc='upper left')
    return [fig1, fig2]

def plot_many_learning_curves(histories, model_names):
    figs, ax = plot_learning_curves(histories[0], model_names[0])
    for history, model_name in zip(histories[1:], model_names[1:]):
        figs = plot_learning_curves(history, model_name, figs=figs, ax=ax)
        