import numpy as np
import math
import time
import ot
import matplotlib.pyplot as plt


def generate_mu(clusters):
    s = 0
    for cluster in clusters:
        s += cluster.shape[0]
    return [1/s]*s

def dist(point1, point2):
    if point1.shape[0] == 2:
        return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
    if point1.shape[0] == 3:
        return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2
    
def generate_d_graph_1(points, adj):
    n = points.shape[0]
    max_dist = 10.0
    d = np.zeros((n, n)) + max_dist
    for src in range(n):
        not_visited = np.array([True] * n)
        shortest_path = np.array([max_dist] * n)
        shortest_path[src] = 0
        while not_visited.sum() > 0:
#             print(shortest_path)
#             print(not_visited)
            current_min = np.argmin((shortest_path - max_dist)*(not_visited))
            if shortest_path[current_min] >= max_dist:
                break
            if not_visited[current_min] == False:
                break
            for neigh in adj[current_min]:
                dis = math.sqrt(dist(points[current_min], points[neigh]))
#                 print(current_min, neigh, dis)
                if shortest_path[current_min] + dis < shortest_path[neigh]:
                    shortest_path[neigh] = shortest_path[current_min] + dis
            not_visited[current_min] = False
        for i in range(n):
            d[src, i] = shortest_path[i]
    return d

def generate_d_graph(clusters, adjs):
    s = 0
    for cluster in clusters:
        s += cluster.shape[0]
    n = s
    ix = np.zeros((n,n))
    d = np.zeros((n,n))
        
    s = 0
    for k in range(len(clusters)):
        dc = generate_d_graph_1(clusters[k], adjs[k])
        for i in range(clusters[k].shape[0]):
            for j in range(clusters[k].shape[0]):
                ix[i+s, j+s] = 1
                d[i+s, j+s] = dc[i, j]**2
        s += clusters[k].shape[0]
    return d, ix

def generate_d(clusters):
    s = 0
    for cluster in clusters:
        s += cluster.shape[0]
    n = s
    ix = np.zeros((n,n))
    d = np.zeros((n,n))
        
    s = 0
    for cluster in clusters:
        for i in range(cluster.shape[0]):
            for j in range(cluster.shape[0]):
                ix[i+s, j+s] = 1
                d[i+s, j+s] = dist(cluster[i], cluster[j])
        s += cluster.shape[0]
    return d, ix

def gamma(mu, d_x, i_x, d_y):
    return np.matmul(np.matmul(i_x, mu), d_y*d_y) - 2*np.matmul(np.matmul(d_x, mu), d_y)

def cost(gamma, mu):
    if mu.shape[0] != gamma.shape[0]:
        print(mu.shape)
        print(gamma.shape)
    s = 0
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            s += mu[i,j] * gamma[i,j]
    return s

def jgw(mu_x, mu_y, d_x, d_y, i_x, epsilon=1e5, eta=1, verbose=False, max_iter=50, diff_thrsh=1e-3, ot_solver='sinkhorn', init=None):
    cur_mu = np.outer(mu_x, mu_y)
    if init is not None:
        cur_mu = init
    dif = []
    costs = []
    mus = []
    for i in range(max_iter):
        t = time.time()
        c = gamma(cur_mu, d_x, i_x, d_y)
        c_o = c
        c -= c.min()
        c = (c**eta) * (cur_mu**(1-eta))       
        k = 1/c.sum() *c.shape[0] * c.shape[1]
        c = c*k
        alpha1=epsilon*k
        if ot_solver == 'sinkhorn':
            mu = ot.sinkhorn(mu_x, mu_y, c, alpha1)
        elif ot_solver == 'sinkhorn_stabilized':
            mu = ot.bregman.sinkhorn_stabilized(mu_x, mu_y, c, alpha1, tau=1e100)
        else:
            mu = ot.emd(mu_x, mu_y, c)

        if verbose:
            print('#####')
            print("iteration: ", i)
            # print(k)
            print("diff: ", abs(mu - cur_mu).sum())
            print("sum: ", mu.sum())
        
        dif.append(abs(mu - cur_mu).sum())
        costs.append(cost(c_o, cur_mu))
    
        if verbose:
            plt.imshow(cur_mu)
            plt.show()
            plt.show()
    
        cur_mu = mu
        mus.append(cur_mu)

        if dif[-1] < diff_thrsh:
            break

        if verbose:
            print("time: ", time.time() - t)
    
    return mus[-1], dif, costs

def jgw_wrapper(points_x, points_y, epsilon=1e5, eta=1, verbose=False, max_iter=50, diff_thrsh=1e-3, ot_solver='sinkhorn', init=None):
    d_y, i_y = generate_d([points_y])
    mu_y = [1/d_y.shape[0]] * d_y.shape[0]
    mu_y = np.array(mu_y)

    d_x, i_x = generate_d(points_x)
    mu_x = [1/d_x.shape[0]] * d_x.shape[0]
    mu_x = np.array(mu_x)

    if verbose:
        plt.imshow(d_x)
        plt.show()
        plt.imshow(d_y)
        plt.show()
        print("d matrices are computed.")

    return jgw(mu_x, mu_y, d_x, d_y, i_x, epsilon=epsilon, eta=eta, verbose=verbose, max_iter=max_iter, diff_thrsh=diff_thrsh, ot_solver=ot_solver, init=init)
