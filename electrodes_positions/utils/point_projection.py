import numpy as np

from utils.geometry import create_rotation_matrices
from utils.geometry import create_rotation_matrix
from utils.insert_points import add_points

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



def project_point_on_triangle(P, vertices):
    # vertices: 3x3 array, each column corresponding to a vertex
    # P: point to project on the triangle

    A = vertices[:,0]
    B = vertices[:,1]
    C = vertices[:,2]

    R = create_rotation_matrix(np.cross(B-A, C-A))


    # this projects point P on the plane spanned by the triangle
    coeffs = np.linalg.inv(R[:2]@np.stack([B-A, C-A]).T)@((R@(P-A))[:2])
    
    # coefficients of the trilinear coordinates that make up the projection on the triangle
    coeffs = np.array([1, 0, 0]) + np.array([[-1, -1], [1,0], [0,1]])@coeffs
    
    # # this is the actual projected point, the formula above is to find directly the trilinear coordinates
    # proj_P = np.linalg.inv(R)@np.array([[1,0,0], [0,1,0], [0,0,0]])@R@(P-A)+A

    # check how many are negative
    neg_coeffs = coeffs<0

    if np.sum(neg_coeffs) == 1:
        # if only one coefficient is negative
        # then closest point is on a segment
        
        start,end = np.nonzero(np.logical_not(neg_coeffs))[0]
        line = vertices[:,end]-vertices[:,start]
        
        # projection of P on the plane
        proj_P = np.sum(coeffs[np.newaxis]*vertices, axis = 1)
        
        # override with new coefficients
        coeffs[end] = np.dot(proj_P-vertices[:,start], line)/np.linalg.norm(line)**2
        coeffs[start] = 1-coeffs[end]
        coeffs[neg_coeffs] *= 0

    coeffs = np.clip(coeffs, 0, 1)
    coeffs /= coeffs.sum()

    # projection of P on the triangle
    proj_P = np.sum(coeffs[np.newaxis]*vertices, axis = 1)

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

def project_pointcloud_on_mesh(points, vertices, faces):
    # projects the point cloud on the input mesh, adding a vertex for each projected point
    points, picked_faces = closest_faces(points, vertices, faces, return_faces=True)
    return add_points(vertices, faces, points, picked_faces)