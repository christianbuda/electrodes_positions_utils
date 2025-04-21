import numpy as np

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

def extract_submesh(vertices, faces, new_vertices):
    # extract submesh from list of new vertices
    
    mask = np.zeros(len(vertices), dtype=bool)
    mask[new_vertices] = True
    faces = faces[mask[faces].all(axis = -1)]
    
    old_vertices = -np.ones(len(vertices), dtype = int)
    old_vertices[new_vertices] = np.arange(len(new_vertices))
    
    new_faces = old_vertices[faces]
    
    assert np.all(new_faces>=0)

    return vertices[new_vertices], new_faces