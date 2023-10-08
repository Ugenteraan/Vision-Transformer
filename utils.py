'''Helper functions.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_accuracy(batch_predictions, batch_targets):
    '''Function to calculate the accuracy of predictions given the targets.
    '''

    return (batch_predictions.argmax(dim=1) == batch_targets.reshape(-1, 1)).float().mean().cpu().numpy()



def plot_loss_acc(path, num_epoch, train_accuracies, train_losses,
                    test_accuracies, test_losses, rank=None):
    '''
    Plot line graphs for the accuracies and loss at every epochs for both training and testing.
    '''

    fig = plt.figure(figsize=(20, 5))
    plt.clf()

    epochs = [x for x in range(num_epoch+1)]

    train_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies, "Mode":['train']*(num_epoch+1)})
    test_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies, "Mode":['test']*(num_epoch+1)})

    data = pd.concat([train_accuracy_df, test_accuracy_df])

    sns.lineplot(data=data, x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Accuracy Graph')
    if not rank is None:
        plt.savefig(path+f'accuracy_epoch.png')
    else:
        plt.savefig(path+f'accuracy_epoch_rank-{rank}.png')

    plt.clf()


    train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_losses, "Mode":['train']*(num_epoch+1)})
    test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_losses, "Mode":['test']*(num_epoch+1)})

    data = pd.concat([train_loss_df, test_loss_df])

    sns.lineplot(data=data, x='Epochs', y='Loss', hue='Mode')
    plt.title('Loss Graph')

    if not rank is None:
        plt.savefig(path+f'loss_epoch.png')
    else:
        plt.savefig(path+f'loss_epoch_rank-{rank}.png')

    plt.close()

    return None


