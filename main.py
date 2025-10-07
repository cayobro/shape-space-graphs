"""
main.py

Author: Carina Veil
Affiliation: Living Matter Lab, Stanford University
Email: cveil@stanford.edu
GitHub: https://github.com/cayobro
Paper: Shape-Space Graphs: Fast and Collision-Free Path Planning for Soft Robots

Description:
This script loads data and plans a collision-free path for a soft robotic arm
using shape-space graphs and obstacle avoidance based on signed distance functions.
Visualization of the planned shape sequence and activations is included.

Note that this is a simplified example using toy data.
"""


import numpy as np
from src.edge_utils import *
from src.graph_utils import *
from src.sdf_utils import *
from src.plotting_utils import *

'''
Note: We provide a small toy dataset, but you can load your own data here.
We need actuations of dimension (N, n_actuators) and configurations of dimension (N, n_zdiscretization, 3).
'''
# Load toy data:
r, R, gamma, N, P = get_data()


## === Define an obstacle and its node clearance ===
box_center = np.array([-0.02, 0.0, 0.060])
box_half_sizes = np.array([0.015, 0.010, 0.015])
plot_obs = [{"type": "box", "center": box_center, "half_sizes": box_half_sizes, "color": "gray", "alpha": 0.25}]
box = lambda X: sdf_box_aabb(X, center=box_center, half_sizes=box_half_sizes)
obstacles = [box]
scene_sdf = lambda X: sdf_scene(X, obstacles, margin=0.0)   
node_clearance, valid_mask = node_clearance_mask(r, scene_sdf)
sweep_ok = make_edge_sweep_checker(r, scene_sdf)

### === Set the parameters of the edge weights ===
params = {
    "r": r,
    "gamma": gamma,
    "alpha": 1.0,   # weight for geometric distance
    "beta":  1.0,   # weight for activation magnitude
    "delta": 1.0,   # weight for activation smoothness
    'sdf_fn': scene_sdf,  # SDF function (or None, if there are no obstacles)
    "node_clearance": node_clearance  # SDF clearance (or None, if there are no obstacles)
}
w = make_edge_weight(params)

## === Build shape-space graph ===
adj = build_knn_graph(R, k=20, weight_fn=w, valid_mask=valid_mask, collision_ok=sweep_ok)

## === Define a route ===
# Choose some nice shapes
r0 = r[-1,:,:]; r1 = r[1,:,:]
waypoints = [r0, r1, r0]
# Find indices of shapes in our library (Trivial here, because we chose from library, but works for random shapes)
nearest_index_to = make_nearest_index_fn(r)
waypoint_indices = [nearest_index_to(wp) for wp in waypoints]

wp_indices = waypoint_planner(waypoint_indices, adj, params)
gamma_seq = gamma[wp_indices,:]
shape_seq = r[wp_indices,:,:]

## === Plot the result ===
ax = plot_shape_sequence(shapes=shape_seq, waypoints=waypoints, obstacles=plot_obs)
for obs in plot_obs:
    if obs["type"] == "cylinder":
        plot_capped_cylinder(ax, obs["center"], obs["radius"], obs["height"],
                            color=obs.get("color", "orange"), alpha=obs.get("alpha", 0.35))
    elif obs["type"] == "box":
        plot_box_aabb(ax, obs["center"], obs["half_sizes"],
                    color=obs.get("color", "red"), alpha=obs.get("alpha", 0.25))

plot_activation_sequence(activations=gamma_seq)

