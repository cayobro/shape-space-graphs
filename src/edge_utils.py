import numpy as np
import torch
import pandas as pd

torch.manual_seed(0)

def make_shape_dist(r):
    """
    Creates a function to compute the shape distance between two points.

    Parameters:
        r (array-like): A list or array of points.

    Returns:
        shape_dist (function): A function that computes the distance between two points i and j.
    """
    def shape_dist(i, j):
        a = r[i]  # Point i
        b = r[j]  # Point j
        # Compute the mean squared distance between corresponding points in a and b
        return np.sqrt(np.mean(np.sum((a - b)**2, axis=1)))
    return shape_dist


def make_edge_weight(params):
    r = params["r"]
    gamma = params["gamma"]
    alpha = params["alpha"]; beta = params["beta"]; delta = params["delta"]; 
    node_clearance = params.get("node_clearance", None)  

    shape_dist = make_shape_dist(r)  # Create the shape distance function
    
    def w(i, j):
        dij = shape_dist(i, j)  # Shape distance between points i and j
        d_mag = 0.5 * (np.matmul(gamma[i].transpose(), gamma[i]) + np.matmul(gamma[j].transpose(), gamma[j])) 
        d_smooth = np.linalg.norm(gamma[j] - gamma[i])  # Difference in gamma values
        return alpha * dij + beta * d_mag + delta * d_smooth
    return w


def make_edge_sweep_checker(r, scene_sdf, s_list=(0.25, 0.5, 0.75)):
    """
    Creates a function to check if an edge is collision-free.

    Parameters:
        r (array-like): A list or array of points.
        scene_sdf (function): A signed distance field function that returns the distance to the nearest obstacle.
        s_list (tuple): A list of interpolation factors to check along the edge.

    Returns:
        ok (function): A function that checks if an edge is collision-free.
    """
    def ok(i, j):
        # Check intermediate points along the edge
        for s in s_list:
            # Interpolate between points i and j
            interp = (1.0 - s) * r[i] + s * r[j]
            # Check if the interpolated point is in collision
            if scene_sdf(interp).min() < 0.0:
                return False
        return True
    return ok


def metrics_table(metrics_dict, title=None, floatfmt=".3f"):
    """
    Pretty-print path metrics as a pandas DataFrame.

    Parameters
    ----------
    metrics_dict : dict
        Output of path_metrics() or similar dictionary of metrics.
    title : str, optional
        Title string to print above the table.
    floatfmt : str
        Format string for floats, e.g. ".3f".

    Returns
    -------
    df : pandas.DataFrame
    """
    # flatten nested 'per_step' dict if present
    flat = metrics_dict.copy()
    if "per_step" in flat:
        for k, v in flat["per_step"].items():
            flat[f"{k}"] = v
        del flat["per_step"]

    # make DataFrame with one row
    df = pd.DataFrame([flat])

    # format floats nicely
    with pd.option_context('display.float_format', lambda x: f"{x:{floatfmt}}"):
        if title:
            print(f"\n=== {title} ===")
        print(df.T.rename(columns={0: "Value"}))  # transpose for vertical view
    return df

