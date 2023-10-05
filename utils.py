'''Helper functions.
'''


def calculate_accuracy(batch_predictions, batch_targets):
    '''Function to calculate the accuracy of predictions given the targets.
    '''

    return (batch_predictions.argmax(dim=1) == batch_targets).float().mean()





