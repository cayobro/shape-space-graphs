import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_color(color_name):
    """
    Retrieve RGB values of a color specified by its name.

    Parameters:
        color_name (str): The name of the color.

    Returns:
        list: RGB values of the color as a list of floats in the range [0, 1].
    """
    colors = {
        "cardinal": [157/255, 34/255, 53/255],
        "palo": [0/255, 106/255, 82/255],
        "palo verde": [39/255, 153/255, 137/255],
        "olive": [143/255, 153/255, 62/255],
        "bay": [111/255, 162/255, 135/255],
        "sky": [66/255, 152/255, 181/255],
        "lagunita": [0/255, 124/255, 146/255],
        "poppy": [233/255, 131/255, 0],
        "plum": [98/255, 0, 89/255],
        "illuminating": [254/255, 197/255, 29/255],
        "spirited": [224/255, 79/255, 57/255],
        "brick": [101/255, 28/255, 50/255],
        "archway": [93/255, 75/255, 60/255]
    }
    return colors.get(color_name.lower(), [0, 0, 0])  # Default to black if color not found


def plot_capped_cylinder(ax, center, radius, height, color='orange', alpha=0.25, n_theta=60, n_h=20):
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(-height/2.0, height/2.0, n_h)
    Theta, Z = np.meshgrid(theta, z)
    X = center[0] + radius*np.cos(Theta)
    Y = center[1] + radius*np.sin(Theta)
    Z = center[2] + Z
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=alpha, color=color)
    ax.plot(center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta),
            center[2] + height/2.0*np.ones_like(theta), color=color, alpha=0.8)
    ax.plot(center[0] + radius*np.cos(theta), center[1] + radius*np.sin(theta),
            center[2] - height/2.0*np.ones_like(theta), color=color, alpha=0.8)


def plot_box_aabb(ax, center, half_sizes, color='red', alpha=0.20):
    cx, cy, cz = center
    hx, hy, hz = half_sizes
    corners = np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz]
    ])
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    polys = [corners[idx] for idx in faces]
    pc = Poly3DCollection(polys, facecolors=color, edgecolors='k', linewidths=0.3, alpha=alpha)
    ax.add_collection3d(pc)


def plot_shape_sequence(shapes, waypoints, obstacles):
    """
    shapes: list/array of (P,3) centerlines (e.g., a planned path)
    waypoints: optional list of (P,3) curves you explicitly provided to the planner
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- draw path ---
    for t, curve in enumerate(shapes):
        color = get_color('palo')
        ax.plot(curve[:,0], curve[:,1], curve[:,2], color=color, linewidth=2)

    # --- draw obstacles ---
    if obstacles:
        for obs in obstacles:
            if obs["type"] == "cylinder":
                plot_capped_cylinder(ax, obs["center"], obs["radius"], obs["height"],
                                     color=obs.get("color","gray"), alpha=obs.get("alpha",0.35))
            elif obs["type"] == "box":
                plot_box_aabb(ax, obs["center"], obs["half_sizes"],
                              color=obs.get("color","red"), alpha=obs.get("alpha",0.25))

    # --- draw explicit waypoints in different color ---
    if waypoints:
        for wp in waypoints:
            ax.plot(wp[:,0], wp[:,1], wp[:,2], color=get_color('cardinal'), linewidth=2)

    equalize_axes(ax)
    plt.show(block=False)
    return ax


def plot_activation_sequence(activations):  
    """
    Plots the gamma values and optionally highlights waypoints.
    Parameters:
        gammas (array-like): Array of gamma values (N x 3).
        waypoint_indices (list, optional): Indices of waypoints to highlight.
        """
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)  # Create subplots for gamma1, gamma2, gamma3
    ax[0].plot(activations[:,0], label='gamma1', color=get_color('palo'), marker='.')
    ax[1].plot(activations[:,1], label='gamma2', color=get_color('sky'), marker='.')    
    ax[2].plot(activations[:,2], label='gamma3', color=get_color('plum'), marker='.')           
    # Add legends and labels
    for i, a in enumerate(ax):
        a.set_ylabel(f'gamma{i+1}')
        a.grid(True)

    ax[2].set_xlabel('Index')  # Label for the x-axis
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.show(block=False)  # Display the plot


def equalize_axes(ax):
    """
    Equalizes the axes of a 3D plot.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to equalize.
    """
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    x_mid = sum(x_limits) / 2
    y_mid = sum(y_limits) / 2
    z_mid = sum(z_limits) / 2
    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

