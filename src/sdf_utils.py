import numpy as np


def sdf_box_aabb(points, center, half_sizes):
    """
    Computes the signed distance from points to an axis-aligned bounding box (AABB).

    Parameters:
        points (array-like): Array of points to compute the distance for (N x 3).
        center (array-like): Center of the AABB (3,).
        half_sizes (array-like): Half-sizes of the AABB along each axis (3,).

    Returns:
        distances (array-like): Signed distances of the points to the AABB (N,).
    """
    # Compute the difference between points and the box center, adjusted for half-sizes
    q = np.abs(points - center) - half_sizes
    # Compute the distance for points outside the box
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    # Compute the distance for points inside the box
    inside = np.minimum(np.max(q, axis=1), 0.0)
    # Combine inside and outside distances
    return outside + inside


def sdf_box_obb(points, center, half_sizes, R):
    """
    Computes the signed distance from points to an oriented bounding box (OBB).

    Parameters:
        points (array-like): Array of points to compute the distance for (N x 3).
        center (array-like): Center of the OBB (3,).
        half_sizes (array-like): Half-sizes of the OBB along each axis (3,).
        R (array-like): Rotation matrix defining the orientation of the OBB (3 x 3).

    Returns:
        distances (array-like): Signed distances of the points to the OBB (N,).
    """
    # Transform points into the local coordinate system of the OBB
    p_loc = (points - center) @ R
    # Compute the difference between points and the box center, adjusted for half-sizes
    q = np.abs(p_loc) - half_sizes
    # Compute the distance for points outside the box
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    # Compute the distance for points inside the box
    inside = np.minimum(np.max(q, axis=1), 0.0)
    # Combine inside and outside distances
    return outside + inside


def sdf_capped_cylinder(points, center, axis, radius, height):
    """
    Computes the signed distance from points to a capped cylinder.

    Parameters:
        points (array-like): Array of points to compute the distance for (N x 3).
        center (array-like): Center of the cylinder (3,).
        axis (array-like): Axis of the cylinder (3,).
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.

    Returns:
        distances (array-like): Signed distances of the points to the capped cylinder (N,).
    """
    # Compute the vector from the cylinder center to the points
    u = points - center
    # Compute the axial distance (distance along the cylinder's axis)
    axial = np.abs(u @ axis) - height / 2.0
    # Compute the radial vector (distance perpendicular to the cylinder's axis)
    radial_vec = u - np.outer(u @ axis, axis)
    # Compute the radial distance (distance from the cylinder's surface)
    radial = np.linalg.norm(radial_vec, axis=1) - radius
    # Combine radial and axial distances into a single array
    d = np.stack([radial, axial], axis=1)
    # Compute the distance for points outside the cylinder
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    # Compute the distance for points inside the cylinder
    inside = np.minimum(np.max(d, axis=1), 0.0)
    # Combine inside and outside distances
    return outside + inside


def sdf_scene(points, obstacles, margin=0.0):
    """
    Computes the signed distance from points to a scene containing multiple obstacles.

    Parameters:
        points (array-like): Array of points to compute the distance for (N x 3).
        obstacles (list of functions): List of SDF functions for the obstacles in the scene.
        margin (float): Safety margin to subtract from the distances.

    Returns:
        distances (array-like): Signed distances of the points to the scene (N,).
    """
    # Compute the signed distance for each obstacle and stack the results
    vals = np.stack([obs(points) for obs in obstacles], axis=1)
    # Compute the minimum distance to any obstacle and subtract the margin
    return np.min(vals, axis=1) - margin


def node_clearance_mask(r, sdf_scene):
    N, P, _ = r.shape
    pts = r.reshape(-1, 3)
    sdf_vals = sdf_scene(pts).reshape(N, P)
    node_clearance = sdf_vals.min(axis=1)
    valid_mask = (node_clearance >= 0.0)
    return node_clearance, valid_mask