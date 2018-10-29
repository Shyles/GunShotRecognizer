import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch import optim, nn
from torch.autograd import Variable
from sklearn.preprocessing import scale
torch.set_num_threads(4)


def train(dataframe):

    # Process data into usable pandas Series' separating wav-files and their targets
    data = [feats for feats in dataframe['feature']]

    targets = dataframe['label']
    targets = targets == 'gun_shot'
    targets = targets.astype(int)
    targets = targets.tolist()

    # # Replace string valued targets with integer values
    # targets, uniques = pd.factorize(dataframe['label'])
    #
    # # Create a dictionary of the labels for later lookup
    # t_labels = dataframe['label'].tolist()
    # labels = {}
    # n = len(uniques)    # Number of unique labels
    #
    # for i in range(len(targets)):
    #     labels[targets[i]] = t_labels[i]

    # Split the data to training and test sets
    train_data, test_data, train_target, test_target = train_test_split(data, targets, train_size=0.8, test_size=0.2)

    print("Training data size:", len(train_data))
    print("Test data size:", len(test_data))
    print()

    # Transform data arrays to tensors
    train_data = Variable(torch.from_numpy(np.array(train_data))).float()
    train_target = Variable(torch.from_numpy(np.array(train_target))).long()
    test_data = Variable(torch.from_numpy(np.array(test_data))).float()
    test_target = Variable(torch.from_numpy(np.array(test_target))).long()

    # # Turn targets to one-hot arrays
    train_target = torch.zeros(len(train_target), 2).scatter_(1, train_target.view(-1,1), 1)
    test_target = torch.zeros(len(test_target), 2).scatter_(1, test_target.view(-1,1), 1)

    # Save test data and targets
    torch.save(test_data, 'testdata.pt')
    torch.save(test_target, 'testtarget.pt')

    n_feat = 40             # Number of features
    n_hidden = 300         # Number of nodes in hidden layers in first layer
    n_output = 2            # Size of output

    # Initialize weight matrices to normal distribution
    w1 = nn.init.normal_(torch.empty(n_feat, n_hidden))
    b1 = nn.init.normal_(torch.empty(n_hidden, ))
    w2 = nn.init.normal_(torch.empty(n_hidden, n_output))
    b2 = nn.init.normal_(torch.empty(n_output, ))
    weights = [w1, b1, w2, b2]

    # Set require_weights to get gradients from PyTorch
    for index, w in enumerate(weights):
        w = Variable(w, requires_grad=True)
        weights[index] = w

    # Initialize optimizer for gradient descent
    lr = 0.01
    opt = optim.SGD(weights, lr=lr)

    # Fit the weight matrix to data with rng iterations
    rng = 5000
    print("Fitting to data \nLearning rate: %.5f\nTraining iterations: %d\nNumber of hidden nodes: %d\n"
          % (lr, rng, n_hidden))
    # Ctrl-c to break loop and still keep the weights in case of bad choice of rng.
    try:
        for i in range(rng):
            # Initialize gradients to prevent buildup
            opt.zero_grad()

            # Calculate loss of prediction
            train_loss = loss(train_data, train_target, weights)

            # Backpropagate: Compute sum of gradients
            train_loss.backward()

            if i == 0:
                print("Training loss on the first iteration: %.8f" % (train_loss.item()))
            elif (i+1) % 20 == 0:
                print("Training loss on the %dth iteration: %.8f" % (i+1, train_loss.item()))

            # Single optimization step
            opt.step()
    except KeyboardInterrupt:
        pass

    # Loss in test_data
    print('Training loss in test data: %f' % loss(test_data, test_target, weights).item())

    # Save weights for later use with the number of hidden nodes to minimize compatibility problems
    f_name = 'weights.pt'
    torch.save(weights, f_name)


def loss(x, y, weights):
    """
    Euclidean distance between vectors
    :param x: Input vector
    :param y: Correct outputs
    :param weights: Neuron weights
    :return: Mean squared error
    """
    y_pred = model(x, weights).squeeze()
    y_pred = (y_pred-y)**2
    return y_pred.sum()/len(y_pred)


def model(x, weights):
    """
    :param x: Input vector
    :param weights: ANN weights
    :return: Output vector of ANN
    """
    w1, b1, w2, b2 = weights
    return torch.mm(torch.sigmoid(torch.mm(x,w1)+b1),w2)+b2

def main(): # Allow importing
    df = pd.read_pickle("dataset.pkl")

    # Train the model on dataset
    train(df)

    # Load weights and data to test accuracy of model
    wei = torch.load('weights.pt')
    testdata = torch.load('testdata.pt')
    testtarget = torch.load('testtarget.pt')

    # Turn prediction to one hot. Softmax is testing for application
    predi = torch.nn.functional.softmax(model(testdata,wei),dim=1)
    values = torch.argmax(predi,1).long()
    predi = torch.zeros(len(values), 2).scatter_(1, values.view(-1,1), 1)
    # Print accuracy in test set
    print("Model's accuray in test set: %f" % accuracy_score(predi.detach().numpy(), testtarget.detach().numpy()))

if __name__ == "__main__":
    main()
