from scipy.optimize import minimize
from scipy.optimize import nnls
import numpy as np
import torch

def ASM(hat_w, c):
    """
    ref:
    http://ofey.me/papers/Pareto.pdf,
    https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1
    :param hat_w: # (K,)
    :param c: # (K,)
    :return:
    """
    A = np.eye(len(c))
    b = hat_w
    x0, _ = nnls(A, b)

    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) + np.sum(c) - 1}
    bounds = [[0., None] for _ in range(len(hat_w))]
    min_out = minimize(_fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
    new_w = min_out.x + c
    return new_w


def pareto_step(w, c, G):
    """
    ref:http://ofey.me/papers/Pareto.pdf
    K : the number of task
    M : the dim of NN's params
    :param W: # (K,1)
    :param C: # (K,1)
    :param G: # (K,M)
    :return:
    """
    GGT = np.matmul(G, G.T)  # (K, K)
    e = np.ones_like(w)  # (K, 1)
    m_up = np.hstack((GGT, e))  # (K, K+1)
    m_down = np.hstack((e.T, np.zeros((1, 1))))  # (1, K+1)
    M = np.vstack((m_up, m_down))  # (K+1, K+1)
    z = np.vstack((-np.matmul(GGT, c), 1 - np.sum(c)))  # (K+1, 1)
    hat_w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(M.T, M)), M.T), z)  # (K+1, 1)
    hat_w = hat_w[:-1].reshape(-1)  # (K,)
    c = c.reshape(-1)  # (K,)
    new_w = ASM(hat_w, c)
    return new_w


def apply_gradient(model, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)


def pareto_fn(w_list, c_list, model, config, loss_list):
    grads = [[] for _ in range(len(loss_list))]

    for i, loss in enumerate(loss_list):
        for p in model.parameters():
            if p.grad is not None:
                grads[i].append(p.grad.view(-1))
            else:
                grads[i].append(torch.zeros_like(p).cuda(non_blocking=True).view(-1))

        grads[i] = torch.cat(grads[i], dim=-1).cpu().numpy()

    grads = np.concatenate(grads, axis=0).reshape(config.num_tasks, -1)
    weights = w_list.reshape(-1, 1)
    c_mat = c_list.reshape(-1, 1)
    new_w_list = pareto_step(weights, c_mat, grads)

    return new_w_list