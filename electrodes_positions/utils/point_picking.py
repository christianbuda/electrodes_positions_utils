import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import pyvista as pv
import time
import trimesh
import networkx as nx


from utils.mesh_utils import faces_to_pyvista
from utils.point_projection import project_pointcloud_on_pointcloud
from utils.mesh_utils import vertex_normal


def pick_fiducials(vertices, faces):
    # opens an interactive window that lets you pick the four fiducial points on the surface
    # returns the list of points in the same order as they were picked, suggested order is RPA, LPA, NAS, IN
    suggested_order = ['RPA', 'LPA', 'NAS', 'IN']
    
    mesh = pv.PolyData(vertices, faces_to_pyvista(faces))
    curv = mesh.curvature()

    picked_points=[]

    def callback(point):
        picked_points.append(point)
        plotter.add_point_labels([point,], [f'{suggested_order[len(picked_points)-1]}'], render_points_as_spheres = True, point_size = 10)
        if len(picked_points) == 4:
            time.sleep(0.1)
            plotter.clear()
            plotter.close()
        

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='red', scalars = curv, pbr=True, metallic=1, cmap = 'gist_ncar', show_scalar_bar = False)
    plotter.add_text(text = 'Select points using right click, following this order: RPA, LPA, NAS, IN', position = 'lower_left')
    plotter.enable_surface_point_picking(callback, show_message = False)
    plotter.show()
    return picked_points

def project_fid_on_mesh(points, target, *args, return_positions = True, return_indices = False):
    return project_pointcloud_on_pointcloud(np.array(points), target, *args, return_positions = return_positions, return_indices = return_indices)

def pick_closed_path(vertices, faces):
    # opens an interactive window that lets you pick a closed path on the input surface
    # the interactive window stops when the path is closed, but the points are saved all along the selection, so closing the path is not needed to use the function
    
    # radii of the spheres that represent the picked points
    radius0 = np.std(vertices)/20
    radius1 = radius0/2

    # useful function for plotting
    lines_to_pyvista = lambda path: np.concatenate([np.repeat(2, len(path)-1)[:,np.newaxis], np.repeat(np.arange(len(path)), 2)[1:-1].reshape((len(path)-1, 2))], axis = 1)


    #########################################################################
    # create a graph from the mesh to allow fast shortest path computations
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # edges without duplication
    edges = mesh.edges_unique

    # create the corresponding graph to compute shortest line path
    G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)
    #########################################################################
    # create pyvista mesh for plotting
    mesh = pv.PolyData(vertices, faces_to_pyvista(faces))
    curv = mesh.curvature()
    #########################################################################
    
    # output arrays
    picked_points = []
    paths = []
    # actors = []

    def callback(point):
        # global picked_points
        # global paths
        # global actors
        
        # project picked point on the vertices
        closest_idx = np.linalg.norm(point-vertices, axis = 1).argmin()
        
        if len(picked_points)==0:
            picked_points.append(closest_idx)
            
            plotter.add_mesh(pv.Sphere(radius = radius0, center = vertices[closest_idx]), pickable=False, color = 'blue')
            
            # old, not working
            # actors.append([])
            # actors[-1].append(plotter.add_mesh(pv.Sphere(radius = radius0, center = vertices[closest_idx]), pickable=False, color = 'blue'))
            # actors[-1].append(plotter.add_point_labels([vertices[closest_idx]], [f'{len(picked_points)-1}'], render_points_as_spheres = True, point_size = 20, pickable = True))
            # actors[-1].append(plotter.add_mesh(pv.PolyData()))
        else:
            # don't know why it does not work..
            # if closest_idx == picked_points[-1]:
            #     actors[-1][0].SetVisibility(False)
            #     actors[-1][1].SetVisibility(False)
            #     actors.pop(-1)
            #     picked_points.pop(-1)
            #     paths.pop(-1)
                
            if np.linalg.norm(vertices[picked_points[0]]-vertices[closest_idx])<radius0:
                closest_idx = picked_points[0]
            
            picked_points.append(closest_idx)
            paths.append(nx.shortest_path(G, source=picked_points[-2], target=picked_points[-1]))
            mesh_path = pv.PolyData(vertices[paths[-1]], lines = lines_to_pyvista(paths[-1]))
            
            plotter.add_mesh(pv.Sphere(radius = radius1, center = vertices[closest_idx]), pickable=False, color = 'blue')
            plotter.add_mesh(mesh_path, show_edges=True, color = 'red')
            
            # actors.append([])
            # actors[-1].append(plotter.add_mesh(pv.Sphere(radius = radius1, center = vertices[closest_idx]), pickable=False, color = 'blue'))
            # actors[-1].append(plotter.add_mesh(mesh_path, show_edges=True, color = 'red'))
            
            
            if picked_points[-1] == picked_points[0]:
                time.sleep(0.1)
                plotter.clear()
                plotter.close()
        

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='red', scalars = curv, pbr=True, metallic=1, cmap = 'gist_ncar', show_scalar_bar = False)
    plotter.add_text(text = 'Create a closed path around the region of interest, pick path points using right click', position = 'lower_left')
    plotter.enable_surface_point_picking(callback, show_message = False, picker = 'volume', pickable_window = False)
    plotter.show()
    
    return picked_points, paths

def compute_path(points, vertices, faces):
    # computes shortest vertex path on the input mesh, given points along the path

    #########################################################################
    # create a graph from the mesh to allow fast shortest path computations
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # edges without duplication
    edges = mesh.edges_unique

    # create the corresponding graph to compute shortest line path
    G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)
        
    paths = []
    
    for i in range(1, len(points)):
        paths.append(nx.shortest_path(G, source=points[i-1], target=points[i])[:-1])
    
    return np.concatenate(paths)


def select_feasible_positions(vertices, faces, positions, landmarks, *args):
    # outputs the subset of positions from the input list that do not enter the paths given in input
    # paths are given as extra args
    
    # create a graph from the mesh to allow fast shortest path computations
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # edges without duplication
    edges = mesh.edges_unique

    to_be_removed = []
    for outlines in args:
        to_be_removed.append(compute_path(outlines, vertices, faces))

    to_be_removed = np.unique(np.concatenate(to_be_removed))


    # create the corresponding graph to compute shortest line path
    G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)

    for node in to_be_removed:
        G.remove_edges_from(list(zip([node]*len(G.adj[node]), G.neighbors(node))))

    good_el = []
    for idx in tqdm(positions):
        if idx in to_be_removed:
            good_el.append(idx)
        else:
            if nx.has_path(G, landmarks['Cz'], idx):
                good_el.append(idx)
                
    return good_el

def refine_IN_LPA(NAS_idx, IN_idx, RPA_idx, LPA_idx, vertices, faces):
    #### CHECK THE FINAL CHOICE!!!
    # not a great algorithm honestly, maybe it could work better with a finer mesh
    # the tolerance should be tuned!!
    #### USE AT YOUR OWN RISK
    
    # given a choice of NAS, IN, RPA, LPA, refines the selection of IN and LPA
    # by choosing vertices around IN and LPA that make NAS-IN as orthogonal as possible to RPA-LPA
    
    mesh = pv.PolyData(vertices, faces_to_pyvista(faces))
    tol = np.std(mesh.bounds)/10
    
    NAS = vertices[NAS_idx]
    IN = vertices[IN_idx]
    RPA = vertices[RPA_idx]
    LPA = vertices[LPA_idx]

    # possible inions are near the input IN, with some tolerance on the x axis
    candidate_IN = np.nonzero(np.abs(vertices[:,0]-IN[0])<tol)[0]

    # we only want the closest points wrt to the yz axes
    candidate_IN = np.intersect1d(candidate_IN, np.nonzero(np.linalg.norm(vertices[:,1:]-IN[1:], axis = 1)<tol/2)[0])

    # possible lpa are near the input LPA, with some tolerance on the yz axes
    candidate_LPA = np.nonzero(np.linalg.norm(vertices[:,1:]-LPA[1:], axis = 1)<tol)[0]

    # we only want the closest points wrt to the normal of the surface
    proj = np.sum(vertices*vertex_normal(LPA_idx, vertices, faces), axis = 1)   # projection of every vertex to the surface normal in LPA_idx
    candidate_LPA = np.intersect1d(candidate_LPA, np.nonzero(np.abs(proj-proj[LPA_idx])<tol/2)[0])

    xvec = RPA-vertices[candidate_LPA]
    yvec = NAS-vertices[candidate_IN]
    all_dots = xvec@yvec.T
    a,b = np.unravel_index(np.argmin(np.abs(all_dots)), all_dots.shape)

    return candidate_IN[b], candidate_LPA[a]


def optimal_sagittal_plane(NAS, IN, RPA, LPA):
    # computes the best normal for the sagittal NAS-IN plane
    # useful if RPA-LPA is not orthogonal to NAS-IN
    # it maximizes the alignment between the normal to the plane and RPA-LPA while keeping NAS and IN inside the plane
    
    v = NAS - IN
    w = RPA - LPA
    
    # (a,b,c) is the target normal
    
    # impose that (a,b,c) is orthogonal to v
    # since v[1] is nonzero (if the coordinate system is somewhat aligned with the head) we can compute:
    f_b = lambda a,c: -(v[0]*a+v[2]*c)/v[1]
    
    # by imposing that a**2+b**2+c**2=1 and using the relationship above, we can find an equation for c
    # in solving it, we choose the solution corresponding to the biggest a (since w should point in the x direction)
    f_a = lambda c: (-v[0]*v[2]*c+np.sqrt(v[1]**2*(v[1]**2+v[0]**2-np.linalg.norm(v)**2*c**2)))/(v[1]**2+v[0]**2)
    
    # with the constraint above, only a range of values for c is allowed, this is defined by:
    lim = np.sqrt(v[0]**2+v[1]**2)/np.linalg.norm(v)
    lim -= lim/1000  # added for numerical tolerance

    # we then want to minimize the dot product between (a,b,c) and w
    obj = lambda c: -(f_a(c)*w[0]+f_b(f_a(c),c)*w[1]+c*w[2])
    
    # in a perfect situation, v is orthogonal to w and the normal coincides with w, i.e. c = 0, which is our starting point
    res = minimize(obj, x0 = 0, bounds = [(-lim, lim)])
    
    if res.success:
        c = res.x
        return np.array([f_a(c), f_b(f_a(c),c), c])[:,0]
    else:
        raise BaseException('Cannot find maximum alignment, something went horribly wrong, check wether the head is more or less aligned with a RAS system')