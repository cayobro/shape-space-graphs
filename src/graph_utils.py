import numpy as np
import torch
import time
import pickle
import heapq
import sklearn
from src.edge_utils import *


def get_data(base_path='/toy-data/'):
    r = pickle.load(open(base_path + 'shapes.pickle', 'rb'))
    gamma = pickle.load(open(base_path + 'activations.pickle', 'rb'))
    z = np.linspace(0, 0.09, 100)

    # add zero config manually
    r0 = np.zeros((r.shape[1], 3))
    r0[:,2] = z
    r = np.vstack([r, r0.reshape(1, *r0.shape)])        
    gamma = np.vstack([gamma, np.zeros((1, gamma.shape[1]))])
    N, P, _ = r.shape
    R = r.reshape(N, -1)
    return r, R, gamma, N, P


def build_knn_graph(centerlines, k, weight_fn, valid_mask=None, collision_ok=None):
    """
    Builds a symmetric k-NN graph with custom edge weights.
    
    Parameters:
        centerlines (np.ndarray): Array of shape (N, D) with points.
        k (int): Number of nearest neighbors.
        weight_fn (function): Function weight_fn(i, j) returning the edge weight.
        valid_mask (np.ndarray, optional): Boolean mask to include/exclude points.
        collision_ok (function, optional): collision_ok(i, j) returns True if edge allowed.
        
    Returns:
        adj (list of lists): Symmetric adjacency list where adj[i] contains (neighbor, weight).
    """
    N = centerlines.shape[0]
    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)
    
    # Step 1: Compute connectivity matrix with KNeighborsTransformer
    trafo = sklearn.neighbors.KNeighborsTransformer(n_neighbors=k, mode='connectivity', algorithm='auto')
    knn_matrix = trafo.fit_transform(centerlines)
    
    # Step 2: Symmetrize to ensure undirected edges
    knn_sym = knn_matrix.maximum(knn_matrix.T)
    
    # Step 3: Build adjacency list applying custom weights and filters
    adj = [[] for _ in range(N)]
    csr = knn_sym.tocsr()
    
    for i in range(N):
        if not valid_mask[i]:
            continue
        start, end = csr.indptr[i], csr.indptr[i+1]
        neighbors = csr.indices[start:end]
        for j in neighbors:
            if not valid_mask[j]:
                continue
            if collision_ok is not None and not collision_ok(i, j):
                continue
            w = weight_fn(i, j)
            adj[i].append((j, w))
    
    return adj


def make_nearest_index_fn(r):
    """
    Creates a function to find the nearest index to a given sample.

    Parameters:
        r (array-like): A list or array of points.

    Returns:
        nearest_index_to (function): A function that finds the nearest index to a given sample.
    """
    def nearest_index_to(sample):
        # Compute the difference between the sample and all points in r
        diffs = r - sample
        # Compute the mean squared distance for each point
        d = np.sqrt(np.mean(np.sum(diffs**2, axis=2), axis=1))
        # Return the index of the point with the smallest distance
        return int(np.argmin(d))
    return nearest_index_to


def dijkstra_shortest_path(start, goal, adj):
    """
    Finds the shortest path between two nodes in a graph using Dijkstra's algorithm.

    Parameters:
        start (int): The starting node.
        goal (int): The goal node.
        adj (list of lists): The adjacency list representation of the graph.

    Returns:
        path (list): The shortest path from start to goal as a list of node indices.
    """
    tic = time.time()  # Start timing
    N = len(adj)  # Number of nodes in the graph
    dist = [np.inf]*N  # Initialize distances to infinity
    prev = [-1]*N  # Initialize previous nodes to -1 (undefined)
    dist[start] = 0.0  # Distance to the start node is 0
    pq = [(0.0, start)]  # Priority queue for Dijkstra's algorithm (min-heap)

    # Main loop of Dijkstra's algorithm
    while pq:
        d, i = heapq.heappop(pq)  # Get the node with the smallest distance
        if i == goal:  # Stop if the goal node is reached
            break
        if d > dist[i]:  # Skip outdated entries
            continue
        # Iterate over neighbors of the current node
        for j, w in adj[i]:
            nd = d + w  # Compute the new distance to neighbor j
            if nd < dist[j]:  # Update if the new distance is smaller
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd, j))  # Push the neighbor into the priority queue

    # Reconstruct the shortest path from start to goal
    path = []
    i = goal
    while i != -1:  # Follow the previous nodes until the start node is reached
        path.append(int(i))
        i = prev[i]
    toc = time.time()  # End timing
    print(f"Shortest path from {start} to {goal} found in {toc - tic:.3f} seconds.")

    # Check if a valid path was found
    if not path:
        print(f"No path found from {start} to {goal}.")
        return []
    return path[::-1]  # Return the path in the correct order (start to goal)


def waypoint_planner(waypoint_indices, adj, params):
    full_path_indices = []
    for start_idx, goal_idx in zip(waypoint_indices[:-1], waypoint_indices[1:]):
        segment = dijkstra_shortest_path(start_idx, goal_idx, adj)
        if full_path_indices:
            # Avoid repeating the first node of the segment
            full_path_indices.extend(segment[1:])
        else:
            full_path_indices.extend(segment)

    m = metrics(full_path_indices, params)
    metrics_table(m, title="Path metrics")

    return full_path_indices


def metrics(path_idx, params):
    gamma = params["gamma"]
    r     = params["r"]
    shape_dist = make_shape_dist(r)

    rs = r[path_idx]          # (m+1, P, 3)
    gs = gamma[path_idx]      # (m+1, f)
    m  = len(path_idx) - 1    # number of edges

    # --- effort (activation space) ---
    E_l2    = np.sum(np.sum(gs**2, axis=1))

    # --- smoothness (activation space) ---
    TV    = np.sum(np.linalg.norm(np.diff(gs, axis=0), axis=1))

    # --- tip path length (workspace) ---
    tips = rs[:, -1, :]                      # (m+1,3), last point = tip
    tip_segs = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    L_tip    = np.sum(tip_segs)

    # --- shape-space length (RMS metric) ---
    shape_segs = [shape_dist(path_idx[k], path_idx[k+1]) for k in range(m)]
    shape_segs = np.asarray(shape_segs, float)
    L_shape    = float(np.sum(shape_segs))

    out = {
        "num_nodes": int(m+1),
        "E_l2 (quadratic effort = energy)": E_l2,
        "TV (total variation: smoothness of path)": TV,
        "L_tip (workspace tip path length)": L_tip,
         "L_shape (RMS shape-space path length)": L_shape,
    }

    return out

