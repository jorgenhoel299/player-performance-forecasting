import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]  # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"]  # update gate weight (notice the variable name)
    bi = parameters["bi"]  # (notice the variable name)
    Wc = parameters["Wc"]  # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"]  # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"]  # prediction weight
    by = parameters["by"]

    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt
    concat = np.concatenate((a_prev, xt), axis=0).astype(float)

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    # Compute prediction of the LSTM cell
    yt_pred = np.dot(Wy, a_next) + by

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
   parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # Initialize "caches", which will track the list of all the caches
    caches = []

    Wy = parameters[
        'Wy']  # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = np.shape(x)
    n_y, n_a = np.shape(Wy)

    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros((n_a, m))

    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:, :, t]
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # Save the value of the next cell state
        c[:, :, t] = c_next
        # Save the value of the prediction in y
        y[:, :, t] = yt
        # Append the cache into caches
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches


def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    # print(da_next.shape, a_next.T.shape)
    # Retrieve dimensions from xt's and a_next's shape
    n_x, m = xt.shape
    n_a, m = a_next.shape

    # Compute gates related derivatives

    dit = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * cct * (1 - it) * it
    dft = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * c_prev * ft * (1 - ft)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (da_next * ot * (1 - np.tanh(c_next) ** 2) + dc_next) * it * (1 - cct ** 2)

    # Compute parameters related derivatives. Use equations
    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis=0).T)  # or use np.dot(dft, np.hstack([a_prev.T, xt.T]))
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis=0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis=0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis=0).T)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # Gradients for the output layer
    # dWy = np.dot(da_next, a_next.T)
    # dby = np.sum(da_next, axis=1, keepdims=True)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input..
    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(
        parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(
        parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}#, "dWy": dWy, "dby": dby}
    return gradients


def lstm_backward(da, caches, y_diff):
    """
    Implementing the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    # Initialize gradients for the output layer
    dWy = np.dot(np.transpose(y_diff, (0, 2, 1)), a0.T)
    # print(dWy.shape)
    dWy = np.sum(dWy, axis=1)
    dWy = np.zeros(np.shape(dWy))
    # print(dWy.shape)
    dby = 0#np.sum(y_diff)
    # print(dby.shape)
    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients["dxt"]
        dWf += gradients["dWf"]
        dWi += gradients["dWi"]
        dWc += gradients["dWc"]
        dWo += gradients["dWo"]
        dbf += gradients["dbf"]
        dbi += gradients["dbi"]
        dbc += gradients["dbc"]
        dbo += gradients["dbo"]
        # Gradients for the output layer
        # dWy += gradients["dWy"]
        # dby += gradients["dby"]

    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = gradients["da_prev"]

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby": dby}

    return gradients


# FROM GPT
def initialize_parameters(n_x, n_a, n_y, seed):
    """
    Initialize parameters for the LSTM model.

    Arguments:
    n_x -- number of features in the input data
    n_a -- number of hidden units in the LSTM
    n_y -- number of output units

    Returns:
    parameters -- a dictionary containing initialized parameters
    """
    np.random.seed(1)  # For reproducibility

    # Forget gate parameters
    Wf = np.random.randn(n_a, n_a + n_x)
    bf = np.zeros((n_a, 1))

    # Update gate parameters
    Wi = np.random.randn(n_a, n_a + n_x)
    bi = np.zeros((n_a, 1))

    # Memory cell parameters
    Wc = np.random.randn(n_a, n_a + n_x)
    bc = np.zeros((n_a, 1))

    # Output gate parameters
    Wo = np.random.randn(n_a, n_a + n_x)
    bo = np.zeros((n_a, 1))
    # Output layer parameters
    if seed is not None:
        np.random.seed(seed)
    Wy = np.random.randn(n_y, n_a)
    by = np.zeros((n_y, 1))

    # Store parameters in a dictionary
    parameters = {"Wf": Wf, "bf": bf, "Wi": Wi, "bi": bi, "Wc": Wc, "bc": bc, "Wo": Wo, "bo": bo, "Wy": Wy, "by": by}

    return parameters


# From GPT
def compute_cost(y_pred, y_true):
    """
    Compute the mean squared error (MSE) loss between predicted and true values.

    Arguments:
    y_pred -- Predicted values, numpy array of shape (n_y, m, T_x)
    y_true -- True values, numpy array of shape (n_y, m, T_x)

    Returns:
    cost -- Mean squared error loss
    """
    m = y_pred.shape[1]  # Number of examples
    T_x = y_pred.shape[2]  # Sequence length
   # print(m, T_x)
    # Compute mean squared error loss
    cost = (1 / (m * T_x)) * np.sum(np.sum((y_pred - y_true) ** 2, axis=1), axis=1)

    return cost


def compute_cost_gradient(y_pred, y_true, caches):
    """
    Compute the gradient of the mean squared error (MSE) loss with respect to y_pred.

    Arguments:
    y_pred -- Predicted values, numpy array of shape (n_y, m, T_x)
    y_true -- True values, numpy array of shape (n_y, m, T_x)
    caches -- Caches from the forward pass (needed for backward pass), tuple

    Returns:
    da -- Gradient of the MSE loss with respect to the hidden states, numpy array of shape (n_a, m, T_x)
    """
    m = y_pred.shape[1]  # Number of examples
    T_x = y_pred.shape[2]  # Sequence length

    # Compute the gradient of the MSE loss
    cost_gradient = (2 / (m * T_x)) * (y_pred - y_true)

    # Initialize the gradient with respect to the hidden states (da)
    shape = (caches[0][0][0].shape[0], caches[0][0][0].shape[1], y_pred.shape[2])
    da = np.zeros(shape=shape)
    # Backpropagate the gradient through time
    for t in reversed(range(T_x)):
        # Add the gradient of the cost at time t to the overall gradient
        da[:, :, t] = cost_gradient[:, :, t]
    return da


# Training loop with performance plotting
def train_lstm(X_train, Y_train, X_val, Y_val, learning_rate=0.1, num_epochs=15, depth=64, plot=False, random_seed=None):
    n_x, m_train, T_x_train = X_train.shape
    _, m_val, T_x_val = X_val.shape
    n_y = Y_train.shape[0]
    n_a = depth  # Adjust the number of hidden units as needed

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_a, n_y, seed=random_seed)

    # Lists to store the cost values for training and validation
    train_costs = []
    val_costs = []

    for epoch in range(num_epochs):
        # Training set forward pass
        a_train, y_train_pred, c_train, caches_train = lstm_forward(X_train, np.zeros((n_a, m_train)), parameters)
        # Compute training cost
        train_cost = compute_cost(y_train_pred, Y_train)
        train_costs.append(train_cost)

        # Validation set forward pass
        a_val, y_val_pred, _, _ = lstm_forward(X_val, np.zeros((n_a, m_val)), parameters)
        # Compute validation cost
        val_cost = compute_cost(y_val_pred, Y_val)
        val_costs.append(val_cost)

        # Backward pass on training set
        da_train = compute_cost_gradient(y_train_pred, Y_train, caches_train)
        gradients_train = lstm_backward(da_train, caches_train, y_train_pred-Y_train)

        # Update parameters using gradient descent for training set
        for param in parameters:
           # if param not in ['Wy', 'by']:
            parameters[param] -= learning_rate * gradients_train["d" + param]

        # Print costs every few epochs
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Training Cost = {train_cost}, Validation Cost = {val_cost}")

    # Plot the cost over epochs for both training and validation sets
    if plot:
        plt.plot(range(num_epochs), train_costs, label='Training Cost')
        plt.plot(range(num_epochs), val_costs, label='Validation Cost')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Training and Validation Costs Over Time')
        plt.legend()
        plt.show()

    return parameters, train_costs, val_costs

# # test it
# from scoring_players import scorer, exhaustion, players_in_match, player_form, opponent_team_form
#
#
#
# path = 'csvs/Patrick-van-Aanholt/2019-2020/summary.csv'
# path2 = 'csvs/Aaron-Cresswell/2019-2020/summary.csv'
# player_stats, player2_stats = pd.read_csv(path), pd.read_csv(path2)
# player_stats['FPL Score'] = player_stats.apply(scorer, axis=1)
# player2_stats['FPL Score'] = player_stats.apply(scorer, axis=1)
# relevant_columns = ['Round', 'Venue', 'FPL Score']
# relevant = player_stats[player_stats.columns.intersection(relevant_columns)]
# relevant2 = player2_stats[player2_stats.columns.intersection(relevant_columns)]
# frames = [relevant, relevant2]
# for i in range(len(frames)):
#     frames[i]['Round'] = frames[i]['Round'].str[-1]
#     frames[i]['Venue'] = frames[i]['Venue'].replace(r'Home', '1', regex=True)
#     frames[i]['Venue'] = frames[i]['Venue'].replace(r'Away', '0', regex=True)
#     frames[i]['Venue'] = frames[i]['Venue'].astype(float)
#     frames[i] = frames[i][frames[i].Round != 'e']
#     frames[i] = frames[i][frames[i].Round != 'd']
#     frames[i]['Round'] = frames[i]['Round']

# mask = ~relevant['Round'].apply(lambda x: any(char.isalpha() for char in x))
# mask2 = ~relevant2['Round'].apply(lambda x: any(char.isalpha() for char in x))
# relevant = relevant[mask]
# relevant2 = relevant[mask2]
# X = relevant[relevant.columns.intersection(['Round', 'Venue'])].values
# y_1 = relevant['FPL Score'].values
# X_2 = relevant2[relevant2.columns.intersection(['Round', 'Venue'])].values
# y_2 = relevant2['FPL Score'].values
#
# rounds = min(len(y_1), len(y_2))
# X, y = X[:rounds, :], y_1[:rounds]
# X_2, y_2 = X_2[:rounds, :], y_2[:rounds]
# data = np.stack((X, X_2, X))
# data = np.transpose(data, (2, 0, 1)).astype(float)
# y = np.row_stack((y, y_2, y))[np.newaxis, :, :]
# print(data.shape, y.shape)
# train_lstm(X=data, Y=y, learning_rate=0.1, num_epochs=10)

# n_a = 4
# m = data.shape[1]
# a0 = np.zeros((n_a, m))
#
# params = initialize_parameters(data.shape[0], n_a, 1)  # n_y = 1 for regression
#
# a, y_hat, c, caches = lstm_forward(data, a0, params)
# print(y_hat.shape[-1])  # .insert(y_hat.shape[-1], 2))
# # for i, cache in enumerate(caches[0]):
# #     print(i+1)
# #     print(cache.shape)
# da = compute_cost_gradient(y_hat, y, caches)
# print(da.shape)
#
# gradients = lstm_backward(da, caches)
