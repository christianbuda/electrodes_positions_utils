import numpy as np
from .mesh_utils import edge_lengths

def add_internal_points(vertices, faces, sampled_points, sampled_faces):
    # add sampled points and faces to the mesh
    # make sure points are internal to the faces (and not on the boundary)
    # generally faster than add_points()
    
    # NOTE: this function can only add one point per triangle at a time!
    if np.any(np.unique(sampled_faces, return_counts = True)[1]>1):
        raise ValueError('add_internal_points() only works when there is a single point per triangle to be added, found more than once here. Try refining the mesh or use add_points().')
    
    sampled_points_idx = np.arange(vertices.shape[0], vertices.shape[0] + sampled_points.shape[0])
    vertices = np.concatenate([vertices, sampled_points])
    
    # add the faces
    # B C D
    t1 = np.concatenate([faces[sampled_faces,1:], sampled_points_idx[:,np.newaxis]], axis = -1)
    # A D C
    t2 = np.concatenate([faces[sampled_faces,:1], sampled_points_idx[:,np.newaxis], faces[sampled_faces,2:]], axis = -1)
    
    faces = np.concatenate([faces, t1, t2])
    
    # replace first face
    faces[sampled_faces,2] = sampled_points_idx
    
    return vertices, faces

def find_close_points(vertices, points):
    # checks if any point in points is very close to a vertex
    
    rtol = 1e-5
    which_close = np.isclose(vertices[np.newaxis],points[:,np.newaxis], rtol = rtol).all(axis = -1)
    
    iter = 0
    while np.any(np.count_nonzero(which_close, axis =-1)>1):
        iter += 1
        
        rtol *= 0.9
        which_close = np.isclose(vertices[np.newaxis],points[:,np.newaxis], rtol = rtol).all(axis = -1)
        
        if iter>1000:
            raise BaseException('There may be a double vertex, check the mesh!')
    
    return which_close

def collinear(A,B,C, epsilon = 1e-2, return_value = False):
    # checks if the 3D points A, B, C are collinear
    # it does so by testing whether the distance between the middle point
    # and the segment between the other two is less than epsilon times smaller than the length of the segment itself
    
    # A, B, C can be arrays of N points, of dimension (N,3)
    
    longest_edge = edge_lengths(A,B,C).max(axis = -1)
    out = np.linalg.norm(np.cross(A-B, A-C), axis = -1)/longest_edge**2
    if not return_value:
        out = out<epsilon/2
    return out


def _add_collinear_point(A,B,C, vertices, faces, sampled_point, sampled_face):
    # adds a point to face [A B C] by splitting edge [A B] a placing the sampled point in the middle
    
    D = vertices.shape[0]
    vertices = np.concatenate([vertices, [sampled_point]])
    
    # new faces (where E is the vertex of the second face to which the edge [A B] belongs)
    # A D C, D B C, A E D, B D E
    
    
    second_face = np.nonzero((np.sum(np.any(faces[np.newaxis] == np.array([A,B])[...,np.newaxis, np.newaxis], axis = 0), axis = -1) == 2)&np.logical_not(np.any(faces == C, axis = -1)))[0]
    E = np.setdiff1d(faces[second_face], np.array([A,B]))[0]
    
    # D B C
    t1 = np.array([[D,B,C]])
    # B D E
    t2 = np.array([[B,D,E]])
    
    faces = np.concatenate([faces, t1, t2])
    faces[sampled_face] = [A,D,C]
    faces[second_face] = [A,E,D]
    
    return vertices, faces, D

def add_single_point(vertices, faces, sampled_point, sampled_face):
    # adds a single point to the mesh
    # if collinear, it adds it by splitting the corresponding edge
    
    # WARNING: you should check if any point is equal to a vertex!!
    
    
    A = faces[sampled_face, 0]
    B = faces[sampled_face, 1]
    C = faces[sampled_face, 2]

    # check if point is collinear to some edge
    if np.any([collinear(vertices[A], vertices[B], sampled_point), collinear(vertices[B], vertices[C], sampled_point), collinear(vertices[C], vertices[A], sampled_point)]):
        which_coll=np.argmin(np.array([collinear(vertices[A], vertices[B], sampled_point, return_value=True), collinear(vertices[B], vertices[C], sampled_point, return_value=True), collinear(vertices[C], vertices[A], sampled_point, return_value=True)]))
        
        # adds the point to the closest edge
        if which_coll==0:
            return _add_collinear_point(A,B,C,vertices, faces, sampled_point, sampled_face)
        elif which_coll == 1:
            return _add_collinear_point(B,C,A,vertices, faces, sampled_point, sampled_face)
        elif which_coll==2:
            return _add_collinear_point(C,A,B,vertices, faces, sampled_point, sampled_face)
    
    # if here, the point is not collinear and can be added safely
    return add_internal_points(vertices, faces, sampled_point[np.newaxis], sampled_face[np.newaxis])+(vertices.shape[0],)

def _safe_add_points(vertices, faces, sampled_points, sampled_faces):
    # subroutine of add_points(), to safely add points (duh..)
    
    nfaces = len(faces)
    
    # check if several points need to be added to the same face
    _, unique_indices = np.unique(sampled_faces, return_index  = True)
    
    
    # save position of added points
    added_points = np.zeros(len(sampled_points), dtype = int)+len(vertices)
    added_points[unique_indices] += np.arange(len(unique_indices))
    
    # add only first occurence of points
    vertices, faces = add_internal_points(vertices, faces, sampled_points[unique_indices], sampled_faces[unique_indices])
    
    # select points that need to be projected again
    left_out_points = np.setdiff1d(np.arange(len(sampled_points)), unique_indices)

    if len(left_out_points)>0:
        sampled_points = sampled_points[left_out_points]
        
        faces_to_check = np.concatenate([sampled_faces, np.arange(nfaces, len(faces))])
        sampled_faces = closest_faces(sampled_points, vertices, faces[faces_to_check], return_faces=True)[1]
        sampled_faces = faces_to_check[sampled_faces]
        
        # sampled_facestrue = closest_faces(sampled_points, vertices, faces, return_faces=True)[1]
        
        
        vertices, faces, newly_added_pts =  add_points(vertices, faces, sampled_points, sampled_faces)
        added_points[left_out_points] = newly_added_pts
    
    return vertices, faces, added_points

def compute_collinearity_matrix(vertices, faces, sampled_points, sampled_faces):
    # returns the collinearity matrix of every point
    # i.e. a (len(sampled_points), 3) boolean matrix with the following properties:
    #       - for each row, the number of True values is either one or zero
    #       - if collinearity_matrix[i, 0] is True, then sampled_points[i] is collinear with A[i] and B[i]
    #       - if collinearity_matrix[i, 1] is True, then sampled_points[i] is collinear with B[i] and C[i]
    #       - if collinearity_matrix[i, 2] is True, then sampled_points[i] is collinear with C[i] and A[i]
    
    A = faces[sampled_faces, 0]
    B = faces[sampled_faces, 1]
    C = faces[sampled_faces, 2]

    collinearity_matrix = np.stack([collinear(vertices[A], vertices[B], sampled_points), collinear(vertices[B], vertices[C], sampled_points), collinear(vertices[C], vertices[A], sampled_points)], axis = -1)
    return collinearity_matrix


def add_points(vertices, faces, sampled_points, sampled_faces):
    # adds points to the mesh "safely", i.e. by splitting edges and joining to closest vertex if necessary
    nfaces = len(faces)
    
    which_vertex = np.any(find_close_points(vertices, sampled_points), axis = -1)
    
    collinearity_matrix = compute_collinearity_matrix(vertices, faces, sampled_points, sampled_faces)
    which_collinear = np.any(collinearity_matrix, axis = -1)
    which_collinear[which_vertex] = False
    which_not_collinear = np.logical_not(which_collinear)
    which_not_collinear[which_vertex] = False
    
    
    # add normal_points
    vertices, faces, newly_added_points = _safe_add_points(vertices, faces, sampled_points[which_not_collinear], sampled_faces[which_not_collinear])
    
    # NO, bisogna considerare anche il caso in cui c'è qualche punto che coincide con un vertice esistente prima di uscire
    # if np.count_nonzero(which_collinear) == 0:
    #     print(f'out {len(newly_added_points)}')
    #     return vertices, faces, newly_added_points
    
    # save position of added points
    added_points = np.zeros(len(which_collinear), dtype = int)
    added_points[which_not_collinear] = newly_added_points
    
    # old
    # which_collinear = np.sum(collinearity_matrix, axis = -1) == 1
    # which_vertex = np.sum(collinearity_matrix, axis = -1) == 2
    
    # take care of points very close to vertices
    if which_vertex.sum()>0:
        added_points[which_vertex] = np.argmin(np.linalg.norm(sampled_points[which_vertex][np.newaxis]-vertices[:,np.newaxis], axis = -1), axis = 0)
    
    sampled_points = sampled_points[which_collinear]
    
    # sampled_facestrue = closest_faces(sampled_points, vertices, faces, return_faces=True)[1]
    faces_to_check = np.concatenate([sampled_faces, np.arange(nfaces, len(faces))])
    
    sampled_faces = closest_faces(sampled_points, vertices, faces[faces_to_check], return_faces=True)[1]
    
    sampled_faces = faces_to_check[sampled_faces]
    
    # _, counts = np.unique(sampled_faces, return_counts  = True)
    # if np.any(counts>2):
    #     raise NotImplementedError('More than one collinear point in the same face, try remeshing or implement the logic to handle this.')
    
    
    which_collinear = np.nonzero(which_collinear)[0]
    
    nfaces = len(faces)
    
    # add sampled points and faces to the mesh
    for i in range(len(sampled_points)):
        # project on faces
        faces_to_check = np.concatenate([sampled_faces, np.arange(nfaces, len(faces))])
        sampled_face = closest_faces(sampled_points[[i]], vertices, faces[faces_to_check], return_faces=True)[1][0]
        sampled_face = faces_to_check[sampled_face]
        
        # sampled_facetrue = closest_faces(sampled_points[[i]], vertices, faces, return_faces=True)[1][0]
        vertices, faces, added_pt = add_single_point(vertices, faces, sampled_points[i], sampled_face)
        added_points[which_collinear[i]] = added_pt
    
    return vertices, faces, added_points


def closest_faces(points, vertices, faces, return_faces = False):
    # projects the input points on the mesh and returns the corresponding points and faces
    # if return_faces is True, the index of the face on which the point was projected is returned

    from .point_projection import project_pointcloud_on_faces
    all_proj = project_pointcloud_on_faces(points, vertices, faces)
    picked_faces = np.linalg.norm(points[:,np.newaxis]-all_proj, axis = -1).argmin(axis = -1)
    
    # projected coordinates
    out = all_proj[np.arange(len(points)),picked_faces]
    
    if return_faces:
        out = (out, picked_faces)
    return out
