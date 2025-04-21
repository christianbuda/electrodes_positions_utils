# coregistration utils
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad


def standardize_fiducials(LPA, RPA, NAS, scale_y = False, shear_y = False):
    # computes the affine transformation to apply to make RPA, LPA coincide with the x axis, with positions +-1
    # and to make NAS lie in the xy plane
    
    # if scale_y is True, the y axis is scaled so as to have the y coordinate of NAS equal to 1
    # if shear_y is True, a shear transformation is applied so as to have NAS lie on the y axis

    # traslation
    traslation_vector = -(LPA+RPA)/2

    LPA = LPA + traslation_vector
    RPA = RPA + traslation_vector
    NAS = NAS + traslation_vector

    # rotation to align RPA-LPA to x axis
    R0 = create_rotation_matrix(RPA-LPA, 'x')
    LPA = R0@LPA
    RPA = R0@RPA
    NAS = R0@NAS

    # rotation around x axis to bring NAS in the xy plane
    projNAS = NAS*np.array([0,1,1])
    R1 = create_rotation_matrix(projNAS, 'y')
    LPA = R1@LPA
    RPA = R1@RPA
    NAS = R1@NAS

    # scale all coordinates in such a way that RPA and LPA have norm 1
    R2 = np.eye(3)/((np.linalg.norm(LPA) + np.linalg.norm(RPA))/2)
    LPA = R2@LPA
    RPA = R2@RPA
    NAS = R2@NAS
    
    # scale y axis so that projNAS has norm 1
    R3 = np.eye(3)
    if scale_y:
        R3[1,1]/= NAS[1]
        LPA = R3@LPA
        RPA = R3@RPA
        NAS = R3@NAS

    # shear so that NAS lies on the x axis
    R4 = np.eye(3)
    if shear_y:
        R4[0,1] = -NAS[0]/NAS[1]
        LPA = R4@LPA
        RPA = R4@RPA
        NAS = R4@NAS
    
    # final rotation matrix
    R = R4@R3@R2@R1@R0
    
    # final traslation vector
    traslation_vector = R@traslation_vector
    
    # encode in affine transform
    A = np.eye(4)
    A[:3,:3] = R
    A[:3,3] = traslation_vector
    
    return A


def transform_fiducials(positions, fiducials, scale_y = False, shear_y = False):
    # transform input positions so as to align them to fiducials
    # positions is a dictionary of (names, position)
    # fiducials is a tuple of position ordered as either (RPA, LPA, NAS, IN) or (RPA, LPA, NAS)
    
    if 'NAS' not in positions.keys() or 'RPA' not in positions.keys() or 'LPA' not in positions.keys():
        raise ValueError('Fiducials must be named as NAS, IN, LPA, RPA inside the positions array')
    
    if len(fiducials) == 4:
        RPA, LPA, NAS, _ = fiducials
    elif len(fiducials) == 3:
        RPA, LPA, NAS = fiducials
    else:
        raise ValueError('fiducials is a tuple of position ordered as either (RPA, LPA, NAS, IN) or (RPA, LPA, NAS)')
    
    A = standardize_fiducials(LPA = positions['LPA'], RPA = positions['RPA'], NAS = positions['NAS'], scale_y = scale_y, shear_y = shear_y)
    A = np.linalg.inv(standardize_fiducials(LPA = LPA, RPA = RPA, NAS = NAS, scale_y = scale_y, shear_y = shear_y))@A
    
    apply_affine = lambda x: (A@np.concatenate([x,[1]]))[:3]
    
    return dict(zip(positions.keys(), map(apply_affine, positions.values())))


def project_electrodes_on_mesh(electrode_positions, vertices, faces):
    # projects input electrode positions on the mesh
    
    # unpack electrode positions
    positions = np.array(list(electrode_positions.values()))
    labels = list(electrode_positions.keys())
    
    # project positions on the mesh vertices
    positions = closest_faces(positions, vertices, faces)
    
    return dict(zip(labels, positions))


def dist_point_mesh(P, vertices, faces):
    # distance between point P and input mesh
    all_dists = np.linalg.norm(P - project_point_on_faces(P, vertices, faces), axis = 1)
    return np.min(all_dists)


def avgsqdist_pointcloud_mesh(points, vertices, faces):
    # average squared distance between each point of the input point cloud and the input mesh
    all_dists = np.sum((points[:,np.newaxis] - project_pointcloud_on_faces(points, vertices, faces))**2, axis = -1)
    all_dists = np.min(all_dists, axis = 1)
    return np.sum(all_dists)/len(points)


def avgsqdist_pointcloud_pointcloud(points, vertices, faces):
    # average squared distance between each point of the input point clouds
    all_dists = np.sum((points - project_pointcloud_on_pointcloud(points, vertices, faces))**2, axis = -1)
    return np.sum(all_dists)/len(points)


def make_affine_transform(params):
    scale_x, scale_y, scale_z, trasl_x, trasl_y, trasl_z, shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy = params
    trasl_mask_x = np.zeros((3,1))
    trasl_mask_x[0,0] = 1
    trasl_mask_y = np.zeros((3,1))
    trasl_mask_y[1,0] = 1
    trasl_mask_z = np.zeros((3,1))
    trasl_mask_z[2,0] = 1
    
    scale_mask_x = np.zeros((3,3))
    scale_mask_x[0,0] = 1
    scale_mask_y = np.zeros((3,3))
    scale_mask_y[1,1] = 1
    scale_mask_z = np.zeros((3,3))
    scale_mask_z[2,2] = 1
    
    shear_mask_xy = np.zeros((3,3))
    shear_mask_xy[0,1] = 1
    shear_mask_xz = np.zeros((3,3))
    shear_mask_xz[0,2] = 1
    shear_mask_yx = np.zeros((3,3))
    shear_mask_yx[1,0] = 1
    shear_mask_yz = np.zeros((3,3))
    shear_mask_yz[1,2] = 1
    shear_mask_zx = np.zeros((3,3))
    shear_mask_zx[2,0] = 1
    shear_mask_zy = np.zeros((3,3))
    shear_mask_zy[2,1] = 1

    R = np.zeros((3,3))

    R = R + scale_x*scale_mask_x + scale_y*scale_mask_y + scale_z*scale_mask_z + trasl_x*trasl_mask_x + trasl_y*trasl_mask_y + trasl_z*trasl_mask_z + shear_xy*shear_mask_xy + shear_xz*shear_mask_xz + shear_yx*shear_mask_yx + shear_yz*shear_mask_yz + shear_zx*shear_mask_zx + shear_zy*shear_mask_zy
    t = trasl_x*trasl_mask_x + trasl_y*trasl_mask_y + trasl_z*trasl_mask_z
    
    return R,t

def rot_x(theta):
    return np.array([[1,0,0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta), np.cos(theta)]])

def rot_y(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)], [0,1,0], [-np.sin(theta),0, np.cos(theta)]])

def rot_z(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0], [np.sin(theta),np.cos(theta),0], [0,0,1]])

def make_rigid_transform(params):
    theta_x, theta_y, theta_z, trasl_x, trasl_y, trasl_z = params
    
    trasl_mask_x = np.zeros((3,1))
    trasl_mask_x[0,0] = 1
    trasl_mask_y = np.zeros((3,1))
    trasl_mask_y[1,0] = 1
    trasl_mask_z = np.zeros((3,1))
    trasl_mask_z[2,0] = 1

    R = rot_z(theta_z)@rot_y(theta_y)@rot_x(theta_x)
    t = trasl_x*trasl_mask_x + trasl_y*trasl_mask_y + trasl_z*trasl_mask_z
    
    return R,t


def make_rigid_transform_with_uniform_scaling(params):
    scale, theta_x, theta_y, theta_z, trasl_x, trasl_y, trasl_z = params
    
    trasl_mask_x = np.zeros((3,1))
    trasl_mask_x[0,0] = 1
    trasl_mask_y = np.zeros((3,1))
    trasl_mask_y[1,0] = 1
    trasl_mask_z = np.zeros((3,1))
    trasl_mask_z[2,0] = 1

    R = scale*rot_z(theta_z)@rot_y(theta_y)@rot_x(theta_x)
    t = trasl_x*trasl_mask_x + trasl_y*trasl_mask_y + trasl_z*trasl_mask_z
    
    return R,t


def make_rigid_transform_with_scaling(params):
    scale_x, scale_y, scale_z, theta_x, theta_y, theta_z, trasl_x, trasl_y, trasl_z = params
    
    scale_mask_x = np.zeros((3,3))
    scale_mask_x[0,0] = 1
    scale_mask_y = np.zeros((3,3))
    scale_mask_y[1,1] = 1
    scale_mask_z = np.zeros((3,3))
    scale_mask_z[2,2] = 1
    
    trasl_mask_x = np.zeros((3,1))
    trasl_mask_x[0,0] = 1
    trasl_mask_y = np.zeros((3,1))
    trasl_mask_y[1,0] = 1
    trasl_mask_z = np.zeros((3,1))
    trasl_mask_z[2,0] = 1

    scale = scale_x*scale_mask_x + scale_y*scale_mask_y + scale_z*scale_mask_z
    
    R = scale@rot_z(theta_z)@rot_y(theta_y)@rot_x(theta_x)
    t = trasl_x*trasl_mask_x + trasl_y*trasl_mask_y + trasl_z*trasl_mask_z
    
    return R,t


def coregister_to_mesh(vertices, faces, electrode_positions, DoF = 7, projection = 'approximate', project_result = True):
    # this functions coregisters and projects the input electrode positions to the input mesh, and returns the coregistered positions
    
    # electrode_positions: dictionary of (names, position)
    # DoF: integer in [6,7,9,12], determines the kind of transformation used in the coregistration
    #      DoF = 6:  rigid transformations
    #      DoF = 7:  rigid transformations with a global scaling factor
    #      DoF = 9:  rotations, traslations, and scalings
    #      DoF = 12:  affine transformations (i.e. rotations, traslations, scalings, and shears)
    # projection: one of ['approximated', 'exact'], determines the algorithm used to compute the projection of the point cloud on the mesh
    #             'approximated' projects the point cloud on the vertices of the mesh, it takes a few seconds to run
    #             'exact' projects the point cloud exactly on the faces of the mesh, it takes a few minutes to run
    # project_result: whether to project the coregistered positions on the input mesh or not
    
    if DoF not in [6,7,9,12]:
        raise ValueError('DoF must be one of [6,7,9,12]')
    
    if projection not in ['approximate', 'exact']:
        raise ValueError("projection must be one of ['approximate', 'exact']")
    
    if DoF == 6:
        make_trans = make_rigid_transform
        init = np.array([0, 0, 0, 0, 0, 0], dtype = float)
    elif DoF == 7:
        make_trans = make_rigid_transform_with_uniform_scaling
        init = np.array([1, 0, 0, 0, 0, 0, 0], dtype = float)
    elif DoF == 9:
        make_trans = make_rigid_transform_with_scaling
        init = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype = float)
    elif DoF == 12:
        make_trans = make_affine_transform
        init = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = float)
        
    if projection == 'approximate':
        projection_function = avgsqdist_pointcloud_pointcloud
    elif projection == 'exact':
        projection_function = avgsqdist_pointcloud_mesh
    
    # unpack electrode positions
    positions = np.array(list(electrode_positions.values()))
    labels = list(electrode_positions.keys())
    
    # objective function to optimize
    def objective(params):
        R,t = make_trans(params)
        pos = (R@positions.T+t).T
        return projection_function(pos, vertices, faces)
    
    res = minimize(objective, init, jac = grad(objective))
    
    if not res.success:
        raise BaseException(f'Optimization terminated unsuccesfully with status {res.status}, and message "{res.message}"')
    
    # apply optimal transformations to the positions
    R,t = make_trans(res.x)
    positions = (R@positions.T+t).T
    
    if project_result:
        # project positions on the mesh vertices
        positions = closest_faces(positions, vertices, faces)
    
    return dict(zip(labels, positions))



# projection utils
def closest_faces(points, vertices, faces, return_faces = False):
    # projects the input points on the mesh and returns the corresponding points and faces
    # if return_faces is True, the index of the face on which the point was projected is returned
    
    all_proj = project_pointcloud_on_faces(points, vertices, faces)
    picked_faces = np.linalg.norm(points[:,np.newaxis]-all_proj, axis = -1).argmin(axis = -1)
    
    # projected coordinates
    out = all_proj[np.arange(len(points)),picked_faces]
    
    if return_faces:
        out = (out, picked_faces)
    return out

def project_pointcloud_on_faces(points, vertices, faces):
    # this function provides the projection of the input point cloud on each triangle of the input mesh
    # it is unnecessarily convoluted to allow the use of the autograd package to compute gradients

    # (N_faces, N_vertex_per_face, 3)
    vertices_groups = vertices[faces]
    N_faces = faces.shape[0]
    N_points = points.shape[0]

    A = vertices_groups[:,0]
    B = vertices_groups[:,1]
    C = vertices_groups[:,2]

    R = create_rotation_matrices(np.cross(B-A, C-A))[np.newaxis]

    # add starting dimension to allow easier broadcasting for point cloud
    vertices_groups = np.broadcast_to(vertices_groups, (N_points,N_faces,3,3))
    A = A[np.newaxis]
    B = B[np.newaxis]
    C = C[np.newaxis]

    # this projects point P on the plane spanned by each triangle
    coeffs = np.linalg.inv(R[:,:,:2]@np.stack([B-A, C-A], axis = -1))@((R@(points[:,np.newaxis]-A)[...,np.newaxis])[:,:,:2])

    # coefficients of the trilinear coordinates that make up the projection on the triangle
    coeffs = np.array([[[[1], [0], [0]]]]) + np.array([[[[-1, -1], [1,0], [0,1]]]])@coeffs

    # # these are the actual projected points on the planes, the formula above is to find directly the trilinear coordinates
    # proj_P = np.squeeze(np.linalg.inv(R)@np.array([[[[1,0,0], [0,1,0], [0,0,0]]]])@R@(points[:,np.newaxis]-A)[..., np.newaxis])+A
    proj_P = np.sum(coeffs*vertices_groups, axis = 2)

    # check how many have exactly one negative coefficient
    pos_coeffs = coeffs[...,0]>0
    which_to_project = np.sum(pos_coeffs, axis = -1) == 2

    # this is a mask on coeffs that is equal to pos_coeffs in points that need to be projected, and is equal to [True, True, False] for all other points
    # it's a trick necessary to avoid item assignment as it is incompatible with autograd
    segments_endpoint = np.where(which_to_project[...,np.newaxis], pos_coeffs, np.ones(pos_coeffs.shape, dtype = bool)*np.array([[[1,1,0]]], dtype = bool))

    # once we have the mask, we can extract the indices
    index_points, index_faces, index_ending = np.nonzero(segments_endpoint)
    index_points = index_points[::2]
    index_faces = index_faces[::2]
    index_starting = index_ending[::2]
    index_ending = index_ending[1::2]

    # and the segments; keep in mind that these segments are only meaningful for the points that need line projection!
    line = vertices_groups[index_points, index_faces, index_ending].reshape((N_points,N_faces,3)) - vertices_groups[index_points, index_faces, index_starting].reshape((N_points,N_faces,3))

    # Use these masks as a trick to avoid item assignment
    # startpoints mask is True on the starting point of each segment (as defined in segments_endpoint)
    startpoints_mask = np.zeros((coeffs.shape))
    startpoints_mask[index_points, index_faces, index_starting] = 1
    # endpoints mask is True on the ending point of each segment (as defined in segments_endpoint)
    endpoints_mask = np.zeros((coeffs.shape))
    endpoints_mask[index_points, index_faces, index_ending] = 1

    # this is the coefficient of the new projected point, relative to the starting vertex
    # the coefficient relative to the ending vertex is its complement to 1 (i.e. 1-startingcoeff)
    tmp = (np.sum((proj_P-vertices_groups[index_points, index_faces, index_starting].reshape((N_points,N_faces,3)))*line, axis = -1)/np.linalg.norm(line, axis = -1)**2)[:,:,np.newaxis, np.newaxis]
    coeffs = (endpoints_mask*tmp + startpoints_mask*(1-tmp))*which_to_project[..., np.newaxis, np.newaxis] + coeffs*np.logical_not(which_to_project)[..., np.newaxis, np.newaxis]

    coeffs = np.clip(coeffs, 0, 1)
    coeffs = coeffs/np.sum(coeffs, axis = -2)[...,np.newaxis]

    proj_P = np.sum(coeffs*vertices_groups, axis = 2)

    return proj_P

def project_point_on_faces(P, vertices, faces):
    # this function provides the projection of point P on each triangle of the input mesh
    # it is unnecessarily convoluted to allow the use of the autograd package to compute gradients

    # (N_faces, N_vertex_per_face, 3)
    vertices_groups = vertices[faces]
    N_faces = faces.shape[0]

    A = vertices_groups[:,0]
    B = vertices_groups[:,1]
    C = vertices_groups[:,2]

    R = create_rotation_matrices(np.cross(B-A, C-A))


    # this projects point P on the plane spanned by each triangle
    coeffs = np.linalg.inv(R[:,:2]@np.stack([B-A, C-A], axis = -1))@((R@(P-A)[...,np.newaxis])[:,:2])

    # coefficients of the trilinear coordinates that make up the projection on the triangle
    coeffs = np.array([[[1], [0], [0]]]) + np.array([[[-1, -1], [1,0], [0,1]]])@coeffs

    # # this is the actual projected point on the plane, the formula above is to find directly the trilinear coordinates
    # proj_P = np.squeeze(np.linalg.inv(R)@np.array([[[1,0,0], [0,1,0], [0,0,0]]])@R@(P-A)[..., np.newaxis])+A
    proj_P = np.sum(coeffs*vertices_groups, axis = 1)

    # check how many have exactly one negative coefficient
    pos_coeffs = coeffs[...,0]>0
    which_to_project = np.sum(pos_coeffs, axis = 1) == 2
    
    # this is a mask on coeffs that is equal to pos_coeffs in points that need to be projected, and is equal to [True, True, False] for all other points
    # it's a trick necessary to avoid item assignment as it's incompatible with autograd
    segments_endpoint = np.where(which_to_project[...,np.newaxis], pos_coeffs, np.ones(pos_coeffs.shape, dtype = bool)*np.array([[1,1,0]], dtype = bool))

    # once we have the mask, we can extract the indices
    segments_endpoint = np.nonzero(segments_endpoint)[1].reshape((N_faces,2))
    
    # and the segments; keep in mind that these segments are only meaningful for the points that need line projection!
    line = vertices_groups[np.arange(N_faces),segments_endpoint[:,1]]-vertices_groups[np.arange(N_faces),segments_endpoint[:,0]]

    # Use these masks as a trick to avoid item assignment
    # startpoints mask is True on the starting point of each segment (as defined in segments_endpoint)
    startpoints_mask = np.zeros((coeffs.shape))
    startpoints_mask[np.arange(N_faces), segments_endpoint[:,0]] = 1
    # endpoints mask is True on the ending point of each segment (as defined in segments_endpoint)
    endpoints_mask = np.zeros((coeffs.shape))
    endpoints_mask[np.arange(N_faces), segments_endpoint[:,1]] = 1
    
    # this is the coefficient of the new projected point, relative to the starting vertex
    # the coefficient relative to the ending vertex is its complement to 1 (i.e. 1-startingcoeff)
    tmp = (np.sum((proj_P-vertices_groups[np.arange(N_faces), segments_endpoint[:,0]])*line, axis = -1)/np.linalg.norm(line, axis = -1)**2)[:,np.newaxis, np.newaxis]
    coeffs = (endpoints_mask*tmp + startpoints_mask*(1-tmp))*which_to_project[..., np.newaxis, np.newaxis] + coeffs*np.logical_not(which_to_project)[..., np.newaxis, np.newaxis]

    coeffs = np.clip(coeffs, 0, 1)
    coeffs = coeffs/np.sum(coeffs, axis = 1)[...,np.newaxis]

    # projection of P on the triangle
    proj_P = np.sum(coeffs*vertices_groups, axis = 1)
    
    return proj_P


def project_pointcloud_on_pointcloud(points, target, *args, return_positions = True, return_indices = False):
    # this function provides the projection of the first input point cloud on the second
    # this is done simply by selecting the closest point in the second set for each point in the first set
    
    # if return_positions is True, the new positions of the projected points are returned
    # if return_indices is True, the indices of the projected points are returned
    

    # compute distances between every couple of points
    proj_P = np.sum((points[:,np.newaxis]-target[np.newaxis])**2, axis = -1)
    proj_P = np.argmin(proj_P, axis = 1)
    
    out = ()
    
    if return_positions:
        out += (target[proj_P],)
    
    if return_indices:
        out += (proj_P,)
    
    return out


# geometry
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