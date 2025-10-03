# Shape-Space Graph: Fast and Collision-Free Path Planning for Soft Robots

This repository contains the code accompanying the paper:  
**Shape-Space Graph: Fast and Collision-Free Path Planning for Soft Robots**  

by Carina Veil, Moritz Flaschel, Ellen Kuhl. 

Contact: cveil@stanford.edu

If you use this code in your research, please cite us: 
---

## Overview
This project presents a graph-based path planning tool for a soft robotic arm.
Using a biomechanical model inspired by morphoelasticity and active filament theory, 
it precomputes a shape library and constructs a k-nearest neighbor graph in shape space, 
where each node corresponds to a mechanically accurate and physically valid robot shape. 
Signed distance functions prune nodes and edges colliding with obstacles, 
while multi-objective edge costs combining geometric distance and actuation effort enable 
energy-efficient planning with collision avoidance. Shortest paths are efficiently found with Dijkstra’s algorithm using the precomputed graph.

---

## Features

- Load toy data of soft robotic arm shapes and activations  
- Define obstacles with flexible SDFs including boxes and cylinders  
- Compute clearance and collision checks for graph edges  
- Build weighted k-nearest neighbor graphs in shape space  
- Plan routes through shape waypoints using Dijkstra’s algorithm  
- Visualize shape sequences, activations, and obstacles in 3D plots  

---

## File Structure

- `main.py`: Entry script to run the path planning
- `edge_utils.py`: Functions to compute shape distances, edge weights, and collision checks  
- `graph_utils.py`: Graph construction, nearest neighbors, shortest path, and waypoint planning  
- `sdf_utils.py`: Signed distance functions for obstacles and scenes  
- `plotting_utils.py`: 3D plotting utilities for shapes, activations, and obstacles  

---

