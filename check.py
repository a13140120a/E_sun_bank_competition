import pickle
import matplotlib.pyplot as plt
import os

def load_pickle_file(pile_path):
    with open(pile_path,'rb') as file:
        pickle_file = pickle.load(file)
    return pickle_file

batch_acc = load_pickle_file('./analyze/batch/acc.pickle')
batch_lr = load_pickle_file('./analyze/batch/lr.pickle')

epoch_acc = load_pickle_file('./analyze/epoch/acc.pickle')
epoch_loss = load_pickle_file('./analyze/epoch/loss.pickle')
epoch_lr = load_pickle_file('./analyze/epoch/lr.pickle')
epoch_val_acc = load_pickle_file('./analyze/epoch/val_acc.pickle')
epoch_val_loss = load_pickle_file('./analyze/epoch/val_loss.pickle')

plt.plot(batch_acc, batch_lr, marker='.')
plt.legend(loc='best')
plt.grid()
plt.xlabel('batch_acc')
plt.ylabel('batch_lr')
plt.show()

