"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def sigmoid(t):
    """
    Applies the sigmoid function elementwise to the input data.

    Parameters
    ----------
    t : array, arbitrary shape
        Input data.

    Returns
    -------
    t_sigmoid : array, arbitrary shape.
        Data after applying the sigmoid function.
    """

    return 1. / (1 + np.exp(-t))


def negative_log_likelihood(X, y, W):
    """
    Negative Log Likelihood of the Logistic Regression.

    Parameters
    ----------
    X : array, shape [N, D]
        (Augmented) feature matrix.
    y : array, shape [N]
        Classification targets.
    w : array, shape [D]
        Regression coefficients (w[0] is the bias term).

    Returns
    -------
    nll : float
        The negative log likelihood.
    """
    return -np.sum(y * np.log(sigmoid(np.dot(X, W))) + (1 - y) * np.log(1 - sigmoid(np.dot(X, W))))


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    num_classes = W.shape[1]
    for c in range(num_classes):
        loss += (1/ len(y))*negative_log_likelihood(X, y==c, W[:, c]) + reg * np.linalg.norm(W[:, c][:-1])**2
        diff = (-((y==c) - sigmoid(np.dot(X, W[:, c]))))
        dW[:, c] = (1 / len(y))*np.dot(diff, X)
        dW[:-1,c] += 2 * reg * np.linalg.norm(W[:, c][:-1])

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################
    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # num_classes = W.shape[1]
    # for c in range(num_classes):
    #     print("NLL", negative_log_likelihood(X, y == c, W[:, c]))
    #     loss += (1 / len(y)) * negative_log_likelihood(X, y == c, W[:, c]) + reg * np.linalg.norm(W[:, c][:-1]) ** 2
    #     dW[:, c] = ((1 / len(y)) *(-((y == c) - sigmoid(X.dot(W[:, c])))).dot(X))
    #     dW[:-1,c] += 2 * reg * np.linalg.norm(W[:, c][:-1])

    num_classes = W.shape[1]
    for c in range(num_classes):
        loss += (1/ len(y))*negative_log_likelihood(X, y==c, W[:, c]) + reg * np.linalg.norm(W[:, c][:-1])**2
        diff = (-((y==c) - sigmoid(np.dot(X, W[:, c]))))
        dW[:, c] = (1 / len(y))*np.dot(diff, X)
        dW[:-1,c] += 2 * reg * W[:-1, c]

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################
    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    #learning_rates = [1e-7, 5e-7]
    learning_rates = [3e-7]
    #regularization_strengths = [2.5e4, 5e4]
    regularization_strengths = [1.2e3]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    for learning_rate in learning_rates:
        for reg in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=learning_rate, reg=reg,
                          num_iters=4000, verbose=False)
            y_train_pred = softmax.predict(X_train)
            train_acc = np.mean(y_train == y_train_pred)
            y_val_pred = softmax.predict(X_val)
            val_acc = np.mean(y_val == y_val_pred)
            results[(learning_rate, reg)] = (train_acc, val_acc)
            all_classifiers.append((softmax, val_acc))

    best_softmax, best_val = max(all_classifiers, key=lambda x: x[1])
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
