import numpy as np
import trimesh

def faces_to_pyvista(faces):
    # converts faces array to pyvista
    return np.hstack((np.full((faces.shape[0], 1), 3), faces))

def edge_lengths(A,B,C):
    # returns the length of the three edges of the triangle defined by the input vertices
    return np.stack([np.linalg.norm(B-A, axis = -1),
                        np.linalg.norm(C-A, axis = -1),
                        np.linalg.norm(B-C, axis = -1)], axis = -1)

def is_plotter_closed(plotter):
    # Function to check if pyvista plotter is closed
    return(plotter.render_window is None)

def vertex_normal(vertex, vertices, faces):
    faces = faces[np.nonzero(np.any(vertex == faces, axis = -1))[0]]
    
    A = vertices[faces[:,0]]
    B = vertices[faces[:,1]]
    C = vertices[faces[:,2]]

    normals = np.cross(B-A, C-A)
    return np.mean(normals, axis = 0)

def extract_submesh(vertices, faces, new_vertices, return_faces = False):
    # extract submesh from list of new vertices
    
    mask = np.zeros(len(vertices), dtype=bool)
    mask[new_vertices] = True
    mask = mask[faces].all(axis = -1)
    orig_faces = np.nonzero(mask)[0]
    faces = faces[orig_faces]
    
    old_vertices = -np.ones(len(vertices), dtype = int)
    old_vertices[new_vertices] = np.arange(len(new_vertices))
    
    new_faces = old_vertices[faces]
    
    assert np.all(new_faces>=0)

    if return_faces:
        return vertices[new_vertices], new_faces, orig_faces
    else:
        return vertices[new_vertices], new_faces

def compute_face_normals(vertices, faces, return_area = False):
    A = vertices[faces[:,0]]
    B = vertices[faces[:,1]]
    C = vertices[faces[:,2]]
    
    
    if not return_area:
        return np.cross(B-A, C-A)/np.linalg.norm(np.cross(B-A, C-A), axis = -1, keepdims=True)
    else:
        norms = np.linalg.norm(np.cross(B-A, C-A), axis = -1, keepdims=True)
        return np.cross(B-A, C-A)/norms, norms[:,0]/2

def compute_vertex_normals(vertices, faces, normalized = True):
    face_normals, face_areas = compute_face_normals(vertices, faces, return_area = True)

    face_weights = trimesh.Trimesh(vertices=vertices, faces=faces).faces_sparse.multiply(face_areas)

    vertex_normals = np.array(np.concatenate([face_weights.multiply(face_normals[:,0]).sum(axis = 1),
    face_weights.multiply(face_normals[:,1]).sum(axis = 1),
    face_weights.multiply(face_normals[:,2]).sum(axis = 1)], axis = 1))

    if normalized:
        return vertex_normals/np.linalg.norm(vertex_normals, axis = 1, keepdims = True)
    else:
        return vertex_normals
