import numpy as np
import math


def _backtracking(loss, grad_loss, w, data):
    """
    Backtracking algorithm for the gradient descent method.
    
    f: function. The function that we want to optimize.
    grad_f: function. The gradient of f(x).
    x: ndarray. The actual iterate x_k.
    """
    X, y = data
    alpha = 1
    c = 0.8
    tau = 0.25
    
    while loss(w - alpha * grad_loss(w, X, y), X, y) > loss(w, X, y) - c * alpha * np.linalg.norm(grad_loss(w, X, y), 2) ** 2:
        alpha = tau * alpha
        
        if alpha < 1e-3:
            break
    return alpha


def gd(loss, grad_loss, w0, data, k_max, tol_loss, tol_w, alpha=None):
    X, y = data
    curr_w, prev_w = w0, np.inf
    curr_k = 0
    grad_x0, curr_grad = grad_loss(w0, X, y), grad_loss(curr_w, X, y)
    history_w = [w0]
    history_loss = [loss(w0, X, y)]
    history_grad = [grad_x0]
    history_err = [np.linalg.norm(grad_x0, 2)]

    while (curr_k < k_max and 
            not (np.linalg.norm(curr_grad, 2) < tol_loss*np.linalg.norm(grad_x0, 2)) and
            not (np.linalg.norm(curr_w - prev_w, 2) < tol_w)):
        if alpha is None:
            alpha = _backtracking(loss, grad_loss, curr_w, data)
        prev_w = curr_w
        curr_w = curr_w - alpha*grad_loss(curr_w, X, y)

        curr_grad = grad_loss(curr_w, X, y)
        curr_k += 1
        
        history_w.append(curr_w)
        history_loss.append(loss(curr_w, X, y))
        history_grad.append(curr_grad)
        history_err.append(np.linalg.norm(curr_grad, 2))

    return history_w, curr_k, history_loss, history_grad, history_err


def sgd(loss, grad_loss, w0, data, batch_size, n_epochs, lr, random_seed=42):
    X, y = data
    data_size = X.shape[1]
    curr_w = w0
    history_w = [w0]
    history_loss = [loss(w0, X, y)]
    history_grad = [grad_loss(w0, X, y)]
    history_err = [np.linalg.norm(history_grad[-1], 2)]

    for _ in range(n_epochs):
        idxs = np.arange(0, data_size)
        np.random.default_rng(random_seed).shuffle(idxs)

        for i in range(math.ceil(data_size / batch_size)):
            batch_idxs = idxs[i*batch_size : (i+1)*batch_size]
            batch_X = X[:, batch_idxs]
            batch_y = y[batch_idxs]

            curr_w = curr_w - lr*grad_loss(curr_w, batch_X, batch_y)

        history_w.append(curr_w)
        history_loss.append(loss(curr_w, X, y))
        history_grad.append(grad_loss(curr_w, X, y))
        history_err.append( np.linalg.norm(history_grad[-1], 2) )

    return history_w, history_loss, history_grad, history_err