import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch import optim, nn
from torch.autograd import Variable
torch.set_num_threads(4)


def classify_gs(dataframe):

    # Process data into usable pandas Series' separating wav-files and their targets
    data = dataframe['data'].values
    data = data.tolist()

    # Replace string valued targets with integer values
    targets, uniques = pd.factorize(dataframe['label'])

    # Create a dictionary of the labels for later lookup
    t_labels = dataframe['label'].tolist()
    labels = {}
    n = len(uniques)    # Number of unique labels

    for i in range(len(targets)):
        labels[targets[i]] = t_labels[i]

    # Split the data to training and test sets
    train_data, test_data, train_target, test_target = train_test_split(data, targets, train_size=0.8, test_size=0.2)

    print("Training data size:", len(train_data))
    print("Test data size:", len(test_data))
    print()

    # Transform data arrays to tensors
    train_data = Variable(torch.from_numpy(np.array(train_data))).float()
    train_target = Variable(torch.from_numpy(np.array(train_target))).float()
    test_data = Variable(torch.from_numpy(np.array(test_data))).float()
    test_target = Variable(torch.from_numpy(np.array(test_target))).float()

    # Turn targets to one-hot arrays
    train_target = torch.zeros(len(train_target), n).scatter_(1, train_target.view(-1,1).long(), 1)
    test_target = torch.zeros(len(test_target), n).scatter_(1, test_target.view(-1,1).long(), 1)

    # Save test data and targets
    torch.save(test_data, 'testdata.pt')
    torch.save(test_target, 'testtarget.pt')

    n_feat = 88375         # Number of features
    n_hidden = 800         # Number of nodes in hidden layer
    n_output = n           # Size of output

    # Initialize weight matrices to normal distribution
    W1 = nn.init.normal_(torch.empty(n_feat, n_hidden))
    b1 = nn.init.normal_(torch.empty(n_hidden, ))
    W2 = nn.init.normal_(torch.empty(n_hidden, n_output))
    b2 = nn.init.normal_(torch.empty(n_output, ))
    weights = [W1, b1, W2, b2]

    # Set require_weights to get gradients from PyTorch
    for index, w in enumerate(weights):
        w = Variable(w, requires_grad=True)
        weights[index] = w

    # Initialize optimizer for gradient descent
    lr = 0.0045
    opt = optim.SGD(weights, lr=lr)

    # Fit the weight matrix to data with rng iterations
    rng = 1600
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
            elif (i+1) % 1 == 0:
                print("Training loss on the %dth iteration: %.8f" % (i+1, train_loss.item()))

            # Single optimization step
            opt.step()
    except KeyboardInterrupt:
        pass

    # Loss in test_data
    print(loss(test_data, test_target, weights).item())

    # Save weights for later use with the number of hidden nodes to minimize compatibility problems
    f_name = 'weights%d.pt' % n_hidden
    torch.save(weights, f_name)

    # Return predictions based on test data, test targets, labels for test targets and the weight matrix
    return model(torch.from_numpy(np.array(test_data)).float(), weights), test_target, labels, weights


def loss(x, y, weights):
    """
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
    W1, b1, W2, b2 = weights
    return torch.mm(torch.sigmoid(torch.mm(x, W1)+b1), W2)+b2


df = pd.read_pickle("dataset.pkl")
classify_gs(df)

w = torch.load('weights800.pt')
testdata = torch.load('testdata.pt')
testtarget = torch.load('testtarget.pt')

_1, x = testtarget.max(-1)
y = model(testdata, w)

m = nn.Softmax(dim=1)
y = m(y)
_, argmax = y.max(-1)
print(argmax)
print(x)
print(accuracy_score(x.detach().numpy(),argmax.detach().numpy()))
