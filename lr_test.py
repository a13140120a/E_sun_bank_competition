from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


# Keras 回調函數，就是追蹤與一個在確定範圍內變化的線性的學習速率相搭配的損失函數。
class LRFinder(Callback):
    def __init__(self, min_lr=0.00001, max_lr=0.001, steps_per_epoch=None, epochs=None,mode='linear'):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        # Record previous batch statistics and update the learning rate.
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def on_train_end(self, logs=None):
        self.plot_lr()
        self.plot_loss()

    def plot_lr(self):
        # Helper function to quickly inspect the learning rate schedule.
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.savefig("../images/images/plot_lr.png")
        plt.show()


    def plot_loss(self):
        # '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['acc'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.savefig("../images/images/plot_loss.png")
        plt.show()


if __name__ == '__main__':
    '''
    min_lr: The lower bound of the learning rate range for the experiment.
    max_lr: The upper bound of the learning rate range for the experiment.
    steps_per_epoch: Number of mini-batches in the dataset.
    epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is 
    '''

    # a=200
    # batch_size = 40
    # epochs = 3
    # lr_finder = LRFinder(min_lr=1e-7, max_lr=1e-4, steps_per_epoch=np.ceil(a // batch_size),
    #                      epochs=epochs)