import numpy as np

def create_rotation_matrix(v, target = 'z'):
    # create a rotation matrix in 3D space in such a way that vector v is rotated along the target direction
    
    v = np.array(v, dtype=float)
    
    v_norm = np.linalg.norm(v)
    
    if v_norm == 0:
        raise ValueError("Zero vector cannot be rotated.")
    
    v = v / v_norm  # Normalize v
    
    if isinstance(target, str):
        if target == 'z':
            target = np.array([0, 0, 1])  # z-axis
        elif target == 'y':
            target = np.array([0, 1, 0])  # z-axis
        elif target == 'x':
            target = np.array([1, 0, 0])  # z-axis
        else:
            raise ValueError("target must be either a vector or one of ['x', 'y', 'z']")
    else:
        target = np.array(target, dtype=float)
    
        target_norm = np.linalg.norm(target)
        
        if np.isclose(target_norm, 0):
            raise ValueError("Zero vector cannot be a target.")
        
        target = target / target_norm  # Normalize target
    
    # If v is already aligned with the z-axis, return identity matrix
    if np.allclose(v, target):
        return np.eye(3)
    
    # Compute rotation axis (cross product of v and target)
    axis = np.cross(v, target)
    axis_norm = np.linalg.norm(axis)
    axis /= axis_norm  # Normalize rotation axis
    
    # Compute rotation angle
    theta = np.arccos(np.dot(v, target)/np.linalg.norm(target))
    
    
    # as described in https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    return R


def create_rotation_matrices(v, target = 'z'):
    # create a set of rotation matrices in 3D space in such a way that the list of vectors v are each rotated along the target direction
    # multidimensional version of "create_rotation_matrix"
    
    v = np.array(v, dtype=float)
    
    v_norm = np.linalg.norm(v, axis = 1)
    
    if np.any(v_norm == 0):
        raise ValueError("Some vectors in the array are null and cannot be rotated.")
    
    v = v / v_norm[...,np.newaxis]  # Normalize v
    
    if isinstance(target, str):
        if target == 'z':
            target = np.array([0, 0, 1])  # z-axis
        elif target == 'y':
            target = np.array([0, 1, 0])  # z-axis
        elif target == 'x':
            target = np.array([1, 0, 0])  # z-axis
        else:
            raise ValueError("target must be either a vector or one of ['x', 'y', 'z']")
    else:
        target = np.array(target, dtype=float)
    
        target_norm = np.linalg.norm(target)
        
        if np.isclose(target_norm, 0):
            raise ValueError("Zero vector cannot be a target.")
        
        target = target / target_norm  # Normalize target
        
    R = np.zeros((v.shape[0], 3, 3))
    
    
    # Compute rotation axes (cross product of v and target)
    axes = np.cross(v, target)
    axes_norm = np.linalg.norm(axes, axis = 1)
    axes /= axes_norm[...,np.newaxis]  # Normalize rotation axis
    
    # Compute rotation angle
    thetas = np.arccos(np.dot(v, target))[...,np.newaxis, np.newaxis]
    
    # as described in https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    R = axes[:, 2, np.newaxis, np.newaxis] * np.array([[[0,-1,0], [1,0,0], [0,0,0]]]) + axes[:, 1, np.newaxis, np.newaxis] * np.array([[[0,0,1], [0,0,0], [-1,0,0]]]) + axes[:, 0, np.newaxis, np.newaxis] * np.array([[[0,0,0], [0,0,-1], [0,1,0]]])
    
    R = np.eye(3)[np.newaxis] + np.sin(thetas) * R + (1 - np.cos(thetas)) * np.matmul(R,R)
    
    return R