import numpy as np
import trimesh
import networkx as nx

from mesh_poisson_disk_sampling import poisson_disk_sampling, uniform_sampling

from .utils.insert_points import add_points
from .utils.geometry import create_rotation_matrix
from .utils.insert_points import closest_faces
from .utils.point_picking import optimal_sagittal_plane
from .utils.point_picking import compute_path
from .utils.mesh_utils import extract_submesh, compute_vertex_normals

_all_montages = {
    '10-20':['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'],
    '10-20-modified':['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'F10', 'T10', 'P10', 'F9', 'T9', 'P9'],
    '10-10-minimal':['Nz','Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz'],
    '10-10':['Nz','Fp1','Fpz','Fp2','AF9','AF7','AF5','AF3','AF1','AFz','AF2','AF4','AF6','AF8','AF10','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10','O1','Oz','O2','I1','Iz','I2'],
    '10-5-reduced':['Nz','Fp1','Fpz','Fp2','AFp3','AFp4','AF9','AF7','AF5h','AF3h','AFz','AF4h','AF6h','AF8','AF10','AFF7h','AFF5h','AFF3h','AFF1h','AFF2h','AFF4h','AFF6h','AFF8h','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FFT9h','FFT7h','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h','FFT8h','FFT10h','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TPP9h','TPP7h','CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','TPP8h','TPP10h','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PPO9h','PPO7h','PPO5h','PPO3h','PPO1h','PPO2h','PPO4h','PPO6h','PPO8h','PPO10h','PO9','PO7','PO5h','PO3h','POz','PO4h','PO6h','PO8','PO10','POO3','POO4','O1','Oz','O2','POO9h','OI1h','OI2h','POO10h','I1','Iz','I2'],
    '10-5-paper':['Nz','Fp1','Fpz','Fp2','AF9','AF7','AF5','AF3','AF1','AFz','AF2','AF4','AF6','AF8','AF10','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10','O1','Oz','O2','I1','Iz','I2','AFp9h','AFp7h','AFp5h','AFp3h','AFp1h','AFp2h','AFp4h','AFp6h','AFp8h','AFp10h','AFF9h','AFF7h','AFF5h','AFF3h','AFF1h','AFF2h','AFF4h','AFF6h','AFF8h','AFF10h','FFT9h','FFT7h','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h','FFT8h','FFT10h','FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h','TTP9h','TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h','TTP10h','TPP9h','TPP7h','CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','TPP8h','TPP10h','PPO9h','PPO7h','PPO5h','PPO3h','PPO1h','PPO2h','PPO4h','PPO6h','PPO8h','PPO10h','POO9h','POO7h','POO5h','POO3h','POO1h','POO2h','POO4h','POO6h','POO8h','POO10h','OI1h','OI2h','Fp1h','Fp2h','AF9h','AF7h','AF5h','AF3h','AF1h','AF2h','AF4h','AF6h','AF8h','AF10h','F9h','F7h','F5h','F3h','F1h','F2h','F4h','F6h','F8h','F10h','FT9h','FT7h','FC5h','FC3h','FC1h','FC2h','FC4h','FC6h','FT8h','FT10h','T9h','T7h','C5h','C3h','C1h','C2h','C4h','C6h','T8h','T10h','TP9h','TP7h','CP5h','CP3h','CP1h','CP2h','CP4h','CP6h','TP8h','TP10h','P9h','P7h','P5h','P3h','P1h','P2h','P4h','P6h','P8h','P10h','PO9h','PO7h','PO5h','PO3h','PO1h','PO2h','PO4h','PO6h','PO8h','PO10h','O1h','O2h','I1h','I2h','AFp9','AFp7','AFp5','AFp3','AFp1','AFpz','AFp2','AFp4','AFp6','AFp8','AFp10','AFF9','AFF7','AFF5','AFF3','AFF1','AFFz','AFF2','AFF4','AFF6','AFF8','AFF10','FFT9','FFT7','FFC5','FFC3','FFC1','FFCz','FFC2','FFC4','FFC6','FFT8','FFT10','FTT9','FTT7','FCC5','FCC3','FCC1','FCCz','FCC2','FCC4','FCC6','FTT8','FTT10','TTP9','TTP7','CCP5','CCP3','CCP1','CCPz','CCP2','CCP4','CCP6','TTP8','TTP10','TPP9','TPP7','CPP5','CPP3','CPP1','CPPz','CPP2','CPP4','CPP6','TPP8','TPP10','PPO9','PPO7','PPO5','PPO3','PPO1','PPOz','PPO2','PPO4','PPO6','PPO8','PPO10','POO9','POO7','POO5','POO3','POO1','POOz','POO2','POO4','POO6','POO8','POO10','OI1','OIz','OI2'],
    '10-5-full':['Nz','Fp1','Fpz','Fp2','AF9','AF7','AF5','AF3','AF1','AFz','AF2','AF4','AF6','AF8','AF10','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10','O1','Oz','O2','I1','Iz','I2','AFp9h','AFp7h','AFp5h','AFp3h','AFp1h','AFp2h','AFp4h','AFp6h','AFp8h','AFp10h','AFF9h','AFF7h','AFF5h','AFF3h','AFF1h','AFF2h','AFF4h','AFF6h','AFF8h','AFF10h','FFT9h','FFT7h','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h','FFT8h','FFT10h','FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h','TTP9h','TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h','TTP10h','TPP9h','TPP7h','CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','TPP8h','TPP10h','PPO9h','PPO7h','PPO5h','PPO3h','PPO1h','PPO2h','PPO4h','PPO6h','PPO8h','PPO10h','POO9h','POO7h','POO5h','POO3h','POO1h','POO2h','POO4h','POO6h','POO8h','POO10h','OI1h','OI2h','Fp1h','Fp2h','AF9h','AF7h','AF5h','AF3h','AF1h','AF2h','AF4h','AF6h','AF8h','AF10h','F9h','F7h','F5h','F3h','F1h','F2h','F4h','F6h','F8h','F10h','FT9h','FT7h','FC5h','FC3h','FC1h','FC2h','FC4h','FC6h','FT8h','FT10h','T9h','T7h','C5h','C3h','C1h','C2h','C4h','C6h','T8h','T10h','TP9h','TP7h','CP5h','CP3h','CP1h','CP2h','CP4h','CP6h','TP8h','TP10h','P9h','P7h','P5h','P3h','P1h','P2h','P4h','P6h','P8h','P10h','PO9h','PO7h','PO5h','PO3h','PO1h','PO2h','PO4h','PO6h','PO8h','PO10h','O1h','O2h','I1h','I2h','AFp9','AFp7','AFp5','AFp3','AFp1','AFpz','AFp2','AFp4','AFp6','AFp8','AFp10','AFF9','AFF7','AFF5','AFF3','AFF1','AFFz','AFF2','AFF4','AFF6','AFF8','AFF10','FFT9','FFT7','FFC5','FFC3','FFC1','FFCz','FFC2','FFC4','FFC6','FFT8','FFT10','FTT9','FTT7','FCC5','FCC3','FCC1','FCCz','FCC2','FCC4','FCC6','FTT8','FTT10','TTP9','TTP7','CCP5','CCP3','CCP1','CCPz','CCP2','CCP4','CCP6','TTP8','TTP10','TPP9','TPP7','CPP5','CPP3','CPP1','CPPz','CPP2','CPP4','CPP6','TPP8','TPP10','PPO9','PPO7','PPO5','PPO3','PPO1','PPOz','PPO2','PPO4','PPO6','PPO8','PPO10','POO9','POO7','POO5','POO3','POO1','POOz','POO2','POO4','POO6','POO8','POO10','OI1','OIz','OI2','FpIz','FpI1','FpI1h','FpI2','FpI2h','N1','N1h','N2','N2h'],
    'all_pos':['Nz','Fp1','Fpz','Fp2','AF9','AF7','AF5','AF3','AF1','AFz','AF2','AF4','AF6','AF8','AF10','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10','O1','Oz','O2','I1','Iz','I2','AFp9h','AFp7h','AFp5h','AFp3h','AFp1h','AFp2h','AFp4h','AFp6h','AFp8h','AFp10h','AFF9h','AFF7h','AFF5h','AFF3h','AFF1h','AFF2h','AFF4h','AFF6h','AFF8h','AFF10h','FFT9h','FFT7h','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h','FFT8h','FFT10h','FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h','TTP9h','TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h','TTP10h','TPP9h','TPP7h','CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','TPP8h','TPP10h','PPO9h','PPO7h','PPO5h','PPO3h','PPO1h','PPO2h','PPO4h','PPO6h','PPO8h','PPO10h','POO9h','POO7h','POO5h','POO3h','POO1h','POO2h','POO4h','POO6h','POO8h','POO10h','OI1h','OI2h','Fp1h','Fp2h','AF9h','AF7h','AF5h','AF3h','AF1h','AF2h','AF4h','AF6h','AF8h','AF10h','F9h','F7h','F5h','F3h','F1h','F2h','F4h','F6h','F8h','F10h','FT9h','FT7h','FC5h','FC3h','FC1h','FC2h','FC4h','FC6h','FT8h','FT10h','T9h','T7h','C5h','C3h','C1h','C2h','C4h','C6h','T8h','T10h','TP9h','TP7h','CP5h','CP3h','CP1h','CP2h','CP4h','CP6h','TP8h','TP10h','P9h','P7h','P5h','P3h','P1h','P2h','P4h','P6h','P8h','P10h','PO9h','PO7h','PO5h','PO3h','PO1h','PO2h','PO4h','PO6h','PO8h','PO10h','O1h','O2h','I1h','I2h','AFp9','AFp7','AFp5','AFp3','AFp1','AFpz','AFp2','AFp4','AFp6','AFp8','AFp10','AFF9','AFF7','AFF5','AFF3','AFF1','AFFz','AFF2','AFF4','AFF6','AFF8','AFF10','FFT9','FFT7','FFC5','FFC3','FFC1','FFCz','FFC2','FFC4','FFC6','FFT8','FFT10','FTT9','FTT7','FCC5','FCC3','FCC1','FCCz','FCC2','FCC4','FCC6','FTT8','FTT10','TTP9','TTP7','CCP5','CCP3','CCP1','CCPz','CCP2','CCP4','CCP6','TTP8','TTP10','TPP9','TPP7','CPP5','CPP3','CPP1','CPPz','CPP2','CPP4','CPP6','TPP8','TPP10','PPO9','PPO7','PPO5','PPO3','PPO1','PPOz','PPO2','PPO4','PPO6','PPO8','PPO10','POO9','POO7','POO5','POO3','POO1','POOz','POO2','POO4','POO6','POO8','POO10','OI1','OIz','OI2','FpIz','FpI1','FpI1h','FpI2','FpI2h','N1','N1h','N2','N2h','LPA','RPA','aboveLPA','aboveRPA', 'T10_line', 'T10h_line', 'T9h_line', 'T9_line'],
    'all':['Nz','Fp1','Fpz','Fp2','AF9','AF7','AF5','AF3','AF1','AFz','AF2','AF4','AF6','AF8','AF10','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8','F10','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T9','T7','C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10','PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10','O1','Oz','O2','I1','Iz','I2','AFp9h','AFp7h','AFp5h','AFp3h','AFp1h','AFp2h','AFp4h','AFp6h','AFp8h','AFp10h','AFF9h','AFF7h','AFF5h','AFF3h','AFF1h','AFF2h','AFF4h','AFF6h','AFF8h','AFF10h','FFT9h','FFT7h','FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h','FFT8h','FFT10h','FTT9h','FTT7h','FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h','FTT10h','TTP9h','TTP7h','CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h','TTP10h','TPP9h','TPP7h','CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h','TPP8h','TPP10h','PPO9h','PPO7h','PPO5h','PPO3h','PPO1h','PPO2h','PPO4h','PPO6h','PPO8h','PPO10h','POO9h','POO7h','POO5h','POO3h','POO1h','POO2h','POO4h','POO6h','POO8h','POO10h','OI1h','OI2h','Fp1h','Fp2h','AF9h','AF7h','AF5h','AF3h','AF1h','AF2h','AF4h','AF6h','AF8h','AF10h','F9h','F7h','F5h','F3h','F1h','F2h','F4h','F6h','F8h','F10h','FT9h','FT7h','FC5h','FC3h','FC1h','FC2h','FC4h','FC6h','FT8h','FT10h','T9h','T7h','C5h','C3h','C1h','C2h','C4h','C6h','T8h','T10h','TP9h','TP7h','CP5h','CP3h','CP1h','CP2h','CP4h','CP6h','TP8h','TP10h','P9h','P7h','P5h','P3h','P1h','P2h','P4h','P6h','P8h','P10h','PO9h','PO7h','PO5h','PO3h','PO1h','PO2h','PO4h','PO6h','PO8h','PO10h','O1h','O2h','I1h','I2h','AFp9','AFp7','AFp5','AFp3','AFp1','AFpz','AFp2','AFp4','AFp6','AFp8','AFp10','AFF9','AFF7','AFF5','AFF3','AFF1','AFFz','AFF2','AFF4','AFF6','AFF8','AFF10','FFT9','FFT7','FFC5','FFC3','FFC1','FFCz','FFC2','FFC4','FFC6','FFT8','FFT10','FTT9','FTT7','FCC5','FCC3','FCC1','FCCz','FCC2','FCC4','FCC6','FTT8','FTT10','TTP9','TTP7','CCP5','CCP3','CCP1','CCPz','CCP2','CCP4','CCP6','TTP8','TTP10','TPP9','TPP7','CPP5','CPP3','CPP1','CPPz','CPP2','CPP4','CPP6','TPP8','TPP10','PPO9','PPO7','PPO5','PPO3','PPO1','PPOz','PPO2','PPO4','PPO6','PPO8','PPO10','POO9','POO7','POO5','POO3','POO1','POOz','POO2','POO4','POO6','POO8','POO10','OI1','OIz','OI2','FpIz','FpI1','FpI1h','FpI2','FpI2h','N1','N1h','N2','N2h','LPA','RPA','aboveLPA','aboveRPA', 'T10_line', 'T10h_line', 'T9h_line', 'T9_line','NAS','IN','O9','O10','T3','T4','T5','T6']
}

available_montages = """'10-20': 19 electrodes according to traditional 10-20 system, see Mecarelli - Clinical EEG, pag. 37, Fig 4.2a\n'10-20-modified': 25 electrodes, recommended as new standard montage (19 as above, plus F10 T10 P10 F9 T9 P9), see Mecarelli - Clinical EEG, pag. 37, Fig 4.2\n'10-10-minimal': 73 electrodes, see Mecarelli - Clinical EEG, pag. 37, Fig 4.2b\n'10-10': 87 electrodes (71 as above, plus AF10 AF9 PO10 PO9 I1 I2 PO3 PO5 PO4 PO6 AF2 AF6 AF3 AF5), see Fig. 1 in  "The five percent electrode system for high-resolution EEG and ERP measurements", in which all except AF10 AF9 are shown\n'10-5-reduced': 145 electrodes selected from the 10-5 system (143 as in Fig. 2 from "The five percent electrode system for high-resolution EEG and ERP measurements", plus AF9 AF10)\n'10-5-paper': 336 electrodes as defined in the 10-5 system, see Fig. 2 in "The five percent electrode system for high-resolution EEG and ERP measurements"\n'10-5-full': 345 electrodes (336 as above, plus FpIz, FpI1, FpI1h, FpI2, FpI2h, N1, N1h, N2, N2h), see "The five percent electrode system for high-resolution EEG and ERP measurements"\n'all_pos': returns all the 353 positions computed (without double names), i.e. the 345 plus LPA, RPA, aboveLPA, aboveRPA, T10_line, T10h_line, T9h_line, T9_line\n'all': returns all the 361 positions (the 353 from above, plus 8 electrodes with a double name: NAS IN O9 O10 T3 T4 T5 T6). No real position is added here, just the second names"""

def reorder_path(path, path_faces, start, end, rtol = 1e-5):
    # reorders path of points of shape (N, 2, 3), i.e. a list of segment termination points
    # into a list of points from start to end with shape (N+1, 3)
    
    # NOTE: start and end must be tuples (i, j) where i indicates the segment and j indicates the terminating point
    # NOTE: face reordered_faces[k] belongs to points (reordered_path[k], reordered_path[k+1])
    
    reordered_path = np.zeros((path.shape[0]+1,3))
    reordered_faces = np.zeros(path.shape[0], dtype = int)
    uninserted = np.ones(path.shape[0], dtype = bool)
    new_indices = np.arange(path.shape[0], dtype = int)

    # initialization
    reordered_path[0] = path[start[0][0], start[1][0]]
    reordered_path[1] = path[start[0][0], (start[1][0]+1)%2]
    reordered_faces[0] = path_faces[start[0][0]]
    reordered_path[-1] = path[end[0][0], end[1][0]]
    reordered_faces[-1] = path_faces[end[0][0]]

    uninserted[start[0][0]] = False
    uninserted[end[0][0]] = False

    for i in range(2,path.shape[0]):
        where_next = np.nonzero(np.all(np.isclose(path[uninserted], reordered_path[i-1], rtol = rtol), axis = -1))
        old_index = new_indices[uninserted][where_next[0][0]]
        reordered_path[i] = path[old_index, (where_next[1][0]+1)%2]
        reordered_faces[i-1] = path_faces[old_index]
        uninserted[old_index] = False

    # consistency check
    assert np.allclose(reordered_path[-2], path[end[0][0], (end[1][0]+1)%2], rtol = rtol), 'Penultimate point mismatch, something went wrong...'
    
    return reordered_path, reordered_faces

def length_percentiles(percentiles, path, path_faces):
    # given an ordered path, this function gives back the points on the path corresponding to the input lengths percentiles
    # the face corresponding to each segment is tracked and returned along with the point

    percentiles = np.array(percentiles)
    assert np.all(percentiles!=0) or np.all(percentiles!=100), 'Percentiles should contain values strictly in (0,100), edge cases should be treated separately.'

    # vectors corresponding to each segment
    segment_vectors = np.diff(path, axis = 0, prepend = [path[0]])

    # length of the segments
    lengths = np.linalg.norm(segment_vectors, axis = -1)

    # cumulative length of the path at point i+1
    dists = np.cumsum(lengths)

    selected_lengths = dists[-1]*percentiles/100

    # index of the last point of the path that the length percentile crosses
    idxs = np.argmax(selected_lengths[...,np.newaxis] <= dists, axis = 1)-1


    # length to cover beyond the last path point
    residual_lengths = selected_lengths - dists[idxs]

    # fraction of the vector to cover
    coefficients = residual_lengths/lengths[idxs+1]


    points = path[idxs] + segment_vectors[idxs+1]*coefficients[...,np.newaxis]
    faces = path_faces[idxs]

    return points, faces

def get_upper_path(vertices, faces, path_plane_normal, start_point, end_point, lower = False):
    # computes the path given the mesh and the plane
    # cuts the lower part of the path by default
    
    # if reverse is True, the lower path is considered
    
    # start_point and end_point should be mesh vertex indices
    
    path_plane_normal = np.squeeze(path_plane_normal)
    
    
    separation_plane = np.cross(path_plane_normal, vertices[end_point]-vertices[start_point])
    
    # check if normal points up
    if lower:
        separation_plane *= -1
    
    # rotate so that input plane contains z axis
    R = create_rotation_matrix(separation_plane, target = 'z')
    vertices = (R@vertices.T).T
    
    assert np.isclose(vertices[end_point, 2], vertices[start_point, 2]), 'Rotation did not produce the correct result, proceeding from here may lead to errors'
    
    plane_center = vertices[[end_point, start_point]].mean(axis = 0)
    
    path, path_faces = trimesh.intersections.mesh_plane(trimesh.Trimesh(vertices=vertices, faces=faces), R@path_plane_normal, plane_center, return_faces=True)

    # remove the part of the path that goes below the endpoints (i.e. the part that loops below the head)
    path_mask = np.all(path[:,:,2]>=vertices[[end_point, start_point],2].min(), axis = -1)

    path = path[path_mask]
    path_faces = path_faces[path_mask]
    
    ############################################################################
    # this section removes lines in the path that start and end at the same point
    # segments that contain the starting point
    start_segments = np.all(np.isclose(path, vertices[start_point]), axis = -1)
    # segments that contain the ending point
    end_segments = np.all(np.isclose(path, vertices[end_point]), axis = -1)
    
    path_mask = np.logical_not(np.all(start_segments, axis = -1)|np.all(end_segments, axis = -1))
    path = path[path_mask]
    path_faces = path_faces[path_mask]
    ############################################################################
    # find rtol such that there is only a single starting point
    rtol = 1e-20
    prev_rtol = 1
    iter = 0
    
    # segments that contain the starting point
    start_segments = np.all(np.isclose(path, vertices[start_point], rtol = rtol), axis = -1)
    
    num_start = start_segments.sum()
    while num_start!=1:
        iter += 1
        
        if num_start > 1:
            if prev_rtol >= rtol:
                prev_rtol = rtol
                rtol /= 10
            else:
                rtol = (prev_rtol + rtol)/2
                prev_rtol = 2*rtol-prev_rtol
        if num_start < 1:
            if prev_rtol <= rtol:
                prev_rtol = rtol
                rtol *= 10
            else:
                rtol = (prev_rtol + rtol)/2
                prev_rtol = 2*rtol-prev_rtol
                
        start_segments = np.all(np.isclose(path, vertices[start_point], rtol = rtol), axis = -1)
        num_start = start_segments.sum()
        if iter > 1000:
            raise BaseException('Cannot find a single starting point! Either try increasing the maximum number of iteration to find the optimal rtol, or check somewhere else for errors.')
    
    # find rtol such that there is only a single ending point
    rtol = 1e-20
    prev_rtol = 1
    iter = 0
    
    # segments that contain the ending point
    end_segments = np.all(np.isclose(path, vertices[end_point]), axis = -1)
    
    num_start = end_segments.sum()
    while num_start!=1:
        iter += 1
        
        if num_start > 1:
            if prev_rtol >= rtol:
                prev_rtol = rtol
                rtol /= 10
            else:
                rtol = (prev_rtol + rtol)/2
                prev_rtol = 2*rtol-prev_rtol
        if num_start < 1:
            if prev_rtol <= rtol:
                prev_rtol = rtol
                rtol *= 10
            else:
                rtol = (prev_rtol + rtol)/2
                prev_rtol = 2*rtol-prev_rtol
                
        end_segments = np.all(np.isclose(path, vertices[start_point], rtol = rtol), axis = -1)
        num_start = end_segments.sum()
        if iter > 1000:
            raise BaseException('Cannot find a single ending point! Either try increasing the maximum number of iteration to find the optimal rtol, or check somewhere else for errors.')
    
    assert start_segments.sum() == 1, f'There should be only one starting point in the path, found {start_segments.sum()} instead'
    assert end_segments.sum() == 1, f'There should be only one ending point in the path, found {end_segments.sum()} instead'
    
    
    where_start = np.nonzero(start_segments)
    where_end = np.nonzero(end_segments)

    # find rtol such that path is consistent
    rtol = 1e-20
    prev_rtol = 1
    restart = True
    iter = 0
    while restart:
        iter += 1
        restart = False
        for i in range(len(path)):
            for j in [0,1]:
                if i == where_start[0][0] and j == where_start[1][0]:
                    continue
                if i == where_end[0][0] and j == where_end[1][0]:
                    continue
                
                num_eq = np.all(np.isclose(path, path[i,j], rtol = 1e-7), axis = -1).sum()
                if num_eq > 2:
                    restart = True
                    if prev_rtol >= rtol:
                        prev_rtol = rtol
                        rtol /= 10
                    else:
                        rtol = (prev_rtol + rtol)/2
                        prev_rtol = 2*rtol-prev_rtol
                elif num_eq < 2:
                    restart = True
                    if prev_rtol <= rtol:
                        prev_rtol = rtol
                        rtol *= 10
                    else:
                        rtol = (prev_rtol + rtol)/2
                        prev_rtol = 2*rtol-prev_rtol
                # print(f'index {i} position {j}, amount {np.all(np.isclose(path, path[i,j]), axis = -1).sum()}')
        if iter > 1000:
            raise BaseException('Path is not well formed, found more than one terminal segment! Either try increasing the maximum number of iteration to find the optimal rtol, or check somewhere else for errors.')
        
    path, path_faces = reorder_path(path, path_faces, start = where_start, end = where_end, rtol = rtol)
    
    path = (np.linalg.inv(R)@path.T).T
    
    return path, path_faces

def find_Cz(vertices, faces, fiducials, epsilon = 1e-5, verbose = False):
    # find Cz as the midpoint between LPA and RPA, and NAS and IN
    # uses an iterative method to find the optimal midpoint

    RPA_idx, LPA_idx, NAS_idx, IN_idx = fiducials
    
    NAS = vertices[NAS_idx]
    IN = vertices[IN_idx]
    RPA = vertices[RPA_idx]
    LPA = vertices[LPA_idx]

    # initialization: Cz as NAS-IN midpoint
    path, path_faces = get_upper_path(vertices, faces, optimal_sagittal_plane(NAS, IN, RPA, LPA), start_point=IN_idx, end_point=NAS_idx)
    Cz, _ = length_percentiles([50], path, path_faces)

    prev_Cz = Cz-100
    iter = 0
    while np.linalg.norm(Cz-prev_Cz)>epsilon:
        iter += 1
        prev_Cz = Cz
        
        if iter%2 == 0:
            plane_normal = np.cross(IN-Cz, NAS-Cz)[0]
            start_point=IN_idx
            end_point=NAS_idx
        else:
            plane_normal = np.cross(RPA-Cz, LPA-Cz)[0]
            start_point=RPA_idx
            end_point=LPA_idx
            
        path, path_faces = get_upper_path(vertices, faces, plane_normal, start_point=start_point, end_point=end_point)
        Cz, Cz_face = length_percentiles([50], path, path_faces)
        
        if verbose:
            print(f'iter {iter}, tol {np.linalg.norm(Cz-prev_Cz)}')
    
    return Cz[0], Cz_face


def path_length(path):
    # takes in input an ordered set of points that form the path
    # and computes the total length
    
    # vectors corresponding to each segment
    segment_vectors = np.diff(path, axis = 0, prepend = [path[0]])

    # length of the segments
    lengths = np.linalg.norm(segment_vectors, axis = -1)
    
    return np.sum(lengths)

def create_standard_montage(vertices, faces, fiducials, system = '10-10', return_indices = False, return_normals = False):
    # this function returns the position of the electrodes placed on the input head according to the input international system
    # possible values are:
    # '10-20': 19 electrodes according to traditional 10-20 system, see Mecarelli - Clinical EEG, pag. 37, Fig 4.2a
    # '10-20-modified': 25 electrodes, recommended as new standard montage (19 as above, plus F10 T10 P10 F9 T9 P9), see Mecarelli - Clinical EEG, pag. 37, Fig 4.2
    # '10-10-minimal': 73 electrodes, see Mecarelli - Clinical EEG, pag. 37, Fig 4.2b
    # '10-10': 87 electrodes (71 as above, plus AF10 AF9 PO10 PO9 I1 I2 PO3 PO5 PO4 PO6 AF2 AF6 AF3 AF5), see Fig. 1 in  "The five percent electrode system for high-resolution EEG and ERP measurements", in which all except AF10 AF9 are shown
    # '10-5-reduced': 145 electrodes selected from the 10-5 system (143 as in Fig. 2 from "The five percent electrode system for high-resolution EEG and ERP measurements", plus AF9 AF10)
    # '10-5-paper': 336 electrodes as defined in the 10-5 system, see Fig. 2 in "The five percent electrode system for high-resolution EEG and ERP measurements"
    # '10-5-full': 345 electrodes (336 as above, plus FpIz, FpI1, FpI1h, FpI2, FpI2h, N1, N1h, N2, N2h), see "The five percent electrode system for high-resolution EEG and ERP measurements"
    # 'all_pos': returns all the 353 positions computed (without double names), i.e. the 345 plus LPA, RPA, aboveLPA, aboveRPA, T10_line, T10h_line, T9h_line, T9_line
    # 'all': returns all the 361 positions (the 353 from above, plus 8 electrodes with a double name: NAS IN O9 O10 T3 T4 T5 T6). No real position is added here, just the second names

    # REFERENCES:
    # https://www.sciencedirect.com/science/article/pii/S1388245700005277?via%3Dihub
    # https://www.acns.org/UserFiles/file/Guideline2-GuidelinesforStandardElectrodePositionNomenclature_v1.pdf
    # Mecarelli - Clinical EEG, chapter 4.2
    # https://robertoostenveld.nl/electrode/#oostenveld2001

    # NOTE on the 10-5 system:
    # in the paper "The five percent electrode system for high-resolution EEG and ERP measurements", they say they defined 345 positions:
    # 336 of these are the positions defined in the image that they show (Fig. 2),
    # the remaining 9 positions can be placed in the holes in front of the head (near the Nz), I call them FpIz, FpI1, FpI1h, FpI2, FpI2h, N1, N1h, N2, N2h
    # they don't mention these names explicitly but they should follow their convention

    # NOTE on the extra positions and the double names:
    # In this code, 353 are defined, which are the 345 from the paper above, plus the RPA, LPA, aboveRPA, aboveLPA, T10_line, T10h_line, T9h_line, T9_line
    # which are only useful to define positions and should not be used to place electrodes.
    # Moreover, 8 electrodes have a double name, according to different conventions, these are the
    # MAS (Nz), IN (Iz), O9 (I1), O10 (I2), T3 (T7), T4 (T8), T5 (P7), T6 (P8)
    # This brings the total number of names defined to 361.

    # NOTE on possible additional positions:
    # According to https://robertoostenveld.nl/electrode/#oostenveld2001, an additional four positions can be defined: A1, A2, M1, M2.
    # A1 and A2 are the references placed on the earlobes, while M1 and M2 are the mastoids. The positions for these electrodes are not well defined
    # (the earlobes are not technically on the scalp, and the mastoids are not mentioned anywhere in the international system definition), so they are not included.
    
    
    global _all_montages
    
    # check validity of system variable
    if system not in _all_montages.keys():
        raise ValueError(f"system must be one of {list(_all_montages.keys())}")
    
    
    (RPA_idx, LPA_idx, NAS_idx, IN_idx) = fiducials

    all_landmarks = {}
    all_landmarks['Nz'] = NAS_idx
    all_landmarks['Iz'] = IN_idx
    all_landmarks['RPA'] = RPA_idx
    all_landmarks['LPA'] = LPA_idx

    Cz, _ = find_Cz(vertices, faces, fiducials=fiducials, epsilon = 1e-5, verbose = False)

    ###################################################################
    # sagittal line through Iz Cz Nz
    names = ['OIz', 'Oz', 'POOz', 'POz', 'PPOz', 'Pz', 'CPPz', 'CPz', 'CCPz', 'Cz', 'FCCz', 'FCz', 'FFCz', 'Fz', 'AFFz', 'AFz', 'AFpz', 'Fpz', 'FpIz']
    start_point = all_landmarks['Iz']
    end_point = all_landmarks['Nz']
    plane_normal = np.cross(vertices[start_point]-Cz, vertices[end_point]-Cz)
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through RPA Cz LPA
    names = ['aboveRPA', 'aboveLPA']
    start_point = all_landmarks['RPA']
    end_point = all_landmarks['LPA']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Cz']], vertices[end_point]-vertices[all_landmarks['Cz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([10,90], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # left transverse line through Oz aboveLPA Fpz
    names = ['O1h', 'O1', 'POO7', 'PO7', 'PPO7', 'P7', 'TPP7', 'TP7', 'TTP7', 'T7', 'FTT7', 'FT7', 'FFT7', 'F7', 'AFF7', 'AF7', 'AFp7', 'Fp1', 'Fp1h']
    start_point = all_landmarks['Oz']
    end_point = all_landmarks['Fpz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['aboveLPA']], vertices[end_point]-vertices[all_landmarks['aboveLPA']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ##################################
    # right transverse line through Oz aboveRPA Fpz
    names = ['O2h', 'O2', 'POO8', 'PO8', 'PPO8', 'P8', 'TPP8', 'TP8', 'TTP8', 'T8', 'FTT8', 'FT8', 'FFT8', 'F8', 'AFF8', 'AF8', 'AFp8', 'Fp2', 'Fp2h']
    start_point = all_landmarks['Oz']
    end_point = all_landmarks['Fpz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['aboveRPA']], vertices[end_point]-vertices[all_landmarks['aboveRPA']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through T8 Cz T7
    # right part
    names = ['T8h', 'C6', 'C6h', 'C4', 'C4h', 'C2', 'C2h']
    start_point = all_landmarks['T8']
    end_point = all_landmarks['T7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Cz']], vertices[end_point]-vertices[all_landmarks['Cz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['Cz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
        
    # left part
    names = ['C1h', 'C1', 'C3h', 'C3', 'C5h', 'C5', 'T7h']
    start_point = all_landmarks['T8']
    end_point = all_landmarks['T7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Cz']], vertices[end_point]-vertices[all_landmarks['Cz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['Cz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    #####################
    # add positions below Oz
    names = ['T10_line', 'T10h_line', 'T9h_line', 'T9_line']
    # store 10% of the length of the previous (upper) path
    ten_percent = path_length(path)/10

    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point, lower = True)

    # compute the corresponding percentiles in the lower path
    ten_percent = 100*ten_percent/path_length(path)
    el_position, el_faces = length_percentiles([ten_percent, ten_percent/2, 100-ten_percent/2, 100-ten_percent], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # left transverse line through OIz T9h FpIz
    names = ['OI1h', 'OI1', 'POO9h', 'PO9h', 'PPO9h', 'P9h', 'TPP9h', 'TP9h', 'TTP9h', 'T9h', 'FTT9h', 'FT9h', 'FFT9h', 'F9h', 'AFF9h', 'AF9h', 'AFp9h', 'FpI1', 'FpI1h']
    start_point = all_landmarks['OIz']
    end_point = all_landmarks['FpIz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['T9h_line']], vertices[end_point]-vertices[all_landmarks['T9h_line']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # right transverse line through OIz T10h FpIz
    names = ['OI2h', 'OI2', 'POO10h', 'PO10h', 'PPO10h', 'P10h', 'TPP10h', 'TP10h', 'TTP10h', 'T10h', 'FTT10h', 'FT10h', 'FFT10h', 'F10h', 'AFF10h', 'AF10h', 'AFp10h', 'FpI2', 'FpI2h']
    start_point = all_landmarks['OIz']
    end_point = all_landmarks['FpIz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['T10h_line']], vertices[end_point]-vertices[all_landmarks['T10h_line']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # left transverse line through Iz T9 Nz
    names = ['I1h', 'I1', 'POO9', 'PO9', 'PPO9', 'P9', 'TPP9', 'TP9', 'TTP9', 'T9', 'FTT9', 'FT9', 'FFT9', 'F9', 'AFF9', 'AF9', 'AFp9', 'N1', 'N1h']
    start_point = all_landmarks['Iz']
    end_point = all_landmarks['Nz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['T9_line']], vertices[end_point]-vertices[all_landmarks['T9_line']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # right transverse line through Iz T10 Nz
    names = ['I2h', 'I2', 'POO10', 'PO10', 'PPO10', 'P10', 'TPP10', 'TP10', 'TTP10', 'T10', 'FTT10', 'FT10', 'FFT10', 'F10', 'AFF10', 'AF10', 'AFp10', 'N2', 'N2h']
    start_point = all_landmarks['Iz']
    end_point = all_landmarks['Nz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['T10_line']], vertices[end_point]-vertices[all_landmarks['T10_line']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through POO8 POOz POO7
    # right part
    names = ['POO8h', 'POO6', 'POO6h', 'POO4', 'POO4h', 'POO2', 'POO2h']
    start_point = all_landmarks['POO8']
    end_point = all_landmarks['POO7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['POOz']], vertices[end_point]-vertices[all_landmarks['POOz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['POOz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['POO1h', 'POO1', 'POO3h', 'POO3', 'POO5h', 'POO5', 'POO7h']
    start_point = all_landmarks['POO8']
    end_point = all_landmarks['POO7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['POOz']], vertices[end_point]-vertices[all_landmarks['POOz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['POOz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through PO8 POz PO7
    # right part
    names = ['PO8h', 'PO6', 'PO6h', 'PO4', 'PO4h', 'PO2', 'PO2h']
    start_point = all_landmarks['PO8']
    end_point = all_landmarks['PO7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['POz']], vertices[end_point]-vertices[all_landmarks['POz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['POz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['PO1h', 'PO1', 'PO3h', 'PO3', 'PO5h', 'PO5', 'PO7h']
    start_point = all_landmarks['PO8']
    end_point = all_landmarks['PO7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['POz']], vertices[end_point]-vertices[all_landmarks['POz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['POz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through PPO8 PPOz PPO7
    # right part
    names = ['PPO8h', 'PPO6', 'PPO6h', 'PPO4', 'PPO4h', 'PPO2', 'PPO2h']
    start_point = all_landmarks['PPO8']
    end_point = all_landmarks['PPO7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['PPOz']], vertices[end_point]-vertices[all_landmarks['PPOz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['PPOz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
        
    # left part
    names = ['PPO1h', 'PPO1', 'PPO3h', 'PPO3', 'PPO5h', 'PPO5', 'PPO7h']
    start_point = all_landmarks['PPO8']
    end_point = all_landmarks['PPO7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['PPOz']], vertices[end_point]-vertices[all_landmarks['PPOz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['PPOz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through P8 Pz P7
    # right part
    names = ['P8h', 'P6', 'P6h', 'P4', 'P4h', 'P2', 'P2h']
    start_point = all_landmarks['P8']
    end_point = all_landmarks['P7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Pz']], vertices[end_point]-vertices[all_landmarks['Pz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['Pz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['P1h', 'P1', 'P3h', 'P3', 'P5h', 'P5', 'P7h']
    start_point = all_landmarks['P8']
    end_point = all_landmarks['P7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Pz']], vertices[end_point]-vertices[all_landmarks['Pz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['Pz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through TPP8 CPPz TPP7
    # right part
    names = ['TPP8h', 'CPP6', 'CPP6h', 'CPP4', 'CPP4h', 'CPP2', 'CPP2h']
    start_point = all_landmarks['TPP8']
    end_point = all_landmarks['TPP7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['CPPz']], vertices[end_point]-vertices[all_landmarks['CPPz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['CPPz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
        
    # left part
    names = ['CPP1h', 'CPP1', 'CPP3h', 'CPP3', 'CPP5h', 'CPP5', 'TPP7h']
    start_point = all_landmarks['TPP8']
    end_point = all_landmarks['TPP7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['CPPz']], vertices[end_point]-vertices[all_landmarks['CPPz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['CPPz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through TP8 CPz TP7
    # right part
    names = ['TP8h', 'CP6', 'CP6h', 'CP4', 'CP4h', 'CP2', 'CP2h']
    start_point = all_landmarks['TP8']
    end_point = all_landmarks['TP7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['CPz']], vertices[end_point]-vertices[all_landmarks['CPz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['CPz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['CP1h', 'CP1', 'CP3h', 'CP3', 'CP5h', 'CP5', 'TP7h']
    start_point = all_landmarks['TP8']
    end_point = all_landmarks['TP7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['CPz']], vertices[end_point]-vertices[all_landmarks['CPz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['CPz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through TTP8 CCPz TTP7
    # right part
    names = ['TTP8h', 'CCP6', 'CCP6h', 'CCP4', 'CCP4h', 'CCP2', 'CCP2h']
    start_point = all_landmarks['TTP8']
    end_point = all_landmarks['TTP7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['CCPz']], vertices[end_point]-vertices[all_landmarks['CCPz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['CCPz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['CCP1h', 'CCP1', 'CCP3h', 'CCP3', 'CCP5h', 'CCP5', 'TTP7h']
    start_point = all_landmarks['TTP8']
    end_point = all_landmarks['TTP7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['CCPz']], vertices[end_point]-vertices[all_landmarks['CCPz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['CCPz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through FTT8 FCCz FTT7
    # right part
    names = ['FTT8h', 'FCC6', 'FCC6h', 'FCC4', 'FCC4h', 'FCC2', 'FCC2h']
    start_point = all_landmarks['FTT8']
    end_point = all_landmarks['FTT7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['FCCz']], vertices[end_point]-vertices[all_landmarks['FCCz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['FCCz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['FCC1h', 'FCC1', 'FCC3h', 'FCC3', 'FCC5h', 'FCC5', 'FTT7h']
    start_point = all_landmarks['FTT8']
    end_point = all_landmarks['FTT7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['FCCz']], vertices[end_point]-vertices[all_landmarks['FCCz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['FCCz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through FT8 FCz FT7
    # right part
    names = ['FT8h', 'FC6', 'FC6h', 'FC4', 'FC4h', 'FC2', 'FC2h']
    start_point = all_landmarks['FT8']
    end_point = all_landmarks['FT7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['FCz']], vertices[end_point]-vertices[all_landmarks['FCz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['FCz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['FC1h', 'FC1', 'FC3h', 'FC3', 'FC5h', 'FC5', 'FT7h']
    start_point = all_landmarks['FT8']
    end_point = all_landmarks['FT7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['FCz']], vertices[end_point]-vertices[all_landmarks['FCz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['FCz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through FFT8 FFCz FFT7
    # right part
    names = ['FFT8h', 'FFC6', 'FFC6h', 'FFC4', 'FFC4h', 'FFC2', 'FFC2h']
    start_point = all_landmarks['FFT8']
    end_point = all_landmarks['FFT7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['FFCz']], vertices[end_point]-vertices[all_landmarks['FFCz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['FFCz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['FFC1h', 'FFC1', 'FFC3h', 'FFC3', 'FFC5h', 'FFC5', 'FFT7h']
    start_point = all_landmarks['FFT8']
    end_point = all_landmarks['FFT7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['FFCz']], vertices[end_point]-vertices[all_landmarks['FFCz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['FFCz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through F8 Fz F7
    # right part
    names = ['F8h', 'F6', 'F6h', 'F4', 'F4h', 'F2', 'F2h']
    start_point = all_landmarks['F8']
    end_point = all_landmarks['F7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Fz']], vertices[end_point]-vertices[all_landmarks['Fz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['Fz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['F1h', 'F1', 'F3h', 'F3', 'F5h', 'F5', 'F7h']
    start_point = all_landmarks['F8']
    end_point = all_landmarks['F7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Fz']], vertices[end_point]-vertices[all_landmarks['Fz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['Fz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through AFF8 AFFz AFF7
    # right part
    names = ['AFF8h', 'AFF6', 'AFF6h', 'AFF4', 'AFF4h', 'AFF2', 'AFF2h']
    start_point = all_landmarks['AFF8']
    end_point = all_landmarks['AFF7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['AFFz']], vertices[end_point]-vertices[all_landmarks['AFFz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['AFFz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['AFF1h', 'AFF1', 'AFF3h', 'AFF3', 'AFF5h', 'AFF5', 'AFF7h']
    start_point = all_landmarks['AFF8']
    end_point = all_landmarks['AFF7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['AFFz']], vertices[end_point]-vertices[all_landmarks['AFFz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['AFFz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through AF8 AFz AF7
    # right part
    names = ['AF8h', 'AF6', 'AF6h', 'AF4', 'AF4h', 'AF2', 'AF2h']
    start_point = all_landmarks['AF8']
    end_point = all_landmarks['AF7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['AFz']], vertices[end_point]-vertices[all_landmarks['AFz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['AFz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['AF1h', 'AF1', 'AF3h', 'AF3', 'AF5h', 'AF5', 'AF7h']
    start_point = all_landmarks['AF8']
    end_point = all_landmarks['AF7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['AFz']], vertices[end_point]-vertices[all_landmarks['AFz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['AFz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # coronal line through AFp8 AFpz AFp7
    # right part
    names = ['AFp8h', 'AFp6', 'AFp6h', 'AFp4', 'AFp4h', 'AFp2', 'AFp2h']
    start_point = all_landmarks['AFp8']
    end_point = all_landmarks['AFp7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['AFpz']], vertices[end_point]-vertices[all_landmarks['AFpz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=all_landmarks['AFpz'])
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    
    # left part
    names = ['AFp1h', 'AFp1', 'AFp3h', 'AFp3', 'AFp5h', 'AFp5', 'AFp7h']
    start_point = all_landmarks['AFp8']
    end_point = all_landmarks['AFp7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['AFpz']], vertices[end_point]-vertices[all_landmarks['AFpz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=all_landmarks['AFpz'], end_point=end_point)
    el_position, el_faces = length_percentiles([12.5, 25, 37.5, 50, 62.5, 75, 87.5], path, path_faces)

    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # add double names
    all_landmarks['O9'] = all_landmarks['I1']
    all_landmarks['O10'] = all_landmarks['I2']
    all_landmarks['T3'] = all_landmarks['T7']
    all_landmarks['T4'] = all_landmarks['T8']
    all_landmarks['T5'] = all_landmarks['P7']
    all_landmarks['T6'] = all_landmarks['P8']
    all_landmarks['NAS'] = all_landmarks['Nz']
    all_landmarks['IN'] = all_landmarks['Iz']

    ###########################################
    
    montage = _all_montages[system]

    if not return_indices:
        montage = {k:vertices[all_landmarks[k]] for k in montage}

        if return_normals:
            all_normals = compute_vertex_normals(vertices, faces, normalized = True)
            normals = {k:all_normals[all_landmarks[k]] for k in montage}
            
            return montage, normals
        else:
            return montage
    else:
        montage = {k:all_landmarks[k] for k in montage}

        return vertices, faces, montage
    
    
def create_custom_montage(vertices, faces, fiducials, subdivisions = None, percentage = None, return_indices = False, return_landmarks = True):
    # creates a custom subdivided montage, starting from the 10-10 system (i.e. subdivisions = 1)
    # subdivisions = 2 coincides with the 10-5 system
    # subdivisions = 3 would be the 10-3.3 system
    # subdivisions = 4 would be the 10-2.5 system
    
    # NOTE: the length percentage can be inserted in place of the subdivisions
    # i.e. percentage = 2.5 is equivalent to subdivisions = 4
    
    # if return_indices is true, a new mesh is returned, and the computed positions are encoded as indices of the mesh
    # if return_landmarks is true, the anatomical landmarks used to compute the positions are returned as well
    
    
    if subdivisions is None:
        assert isinstance(percentage, float)
        if percentage > 10:
            raise ValueError('This montage is based on subdivisions of the 10-10 system, the variable percentage must be a float lower than 10.')

        subdivisions = round(10/percentage)
        
    assert isinstance(subdivisions, int)
    
    # compute 10-5 system
    vertices, faces, all_landmarks = create_standard_montage(vertices, faces, fiducials = fiducials, system = 'all', return_indices = True)
    
    
    all_pos = []

    ###################################################################
    # sagittal line through Iz Cz Nz
    start_point = all_landmarks['Iz']
    end_point = all_landmarks['Nz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Cz']], vertices[end_point]-vertices[all_landmarks['Cz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles(np.linspace(0,100,10*subdivisions, endpoint = False)[1:], path, path_faces)

    vertices, faces, sagittal_midline = add_points(vertices, faces, el_position, el_faces)

    # sagittal_midline = np.arange(vertices.shape[0]-len(el_position), vertices.shape[0])
    sagittal_midline = np.concatenate([[all_landmarks['Iz']], sagittal_midline, [all_landmarks['Nz']]])

    all_landmarks['Oz'] = sagittal_midline[subdivisions]
    all_landmarks['Fpz'] = sagittal_midline[-subdivisions-1]
    all_landmarks['Cz'] = sagittal_midline[5*subdivisions]

    all_pos.append(sagittal_midline)
    ###################################################################
    # coronal line through RPA Cz LPA
    names = ['aboveRPA', 'aboveLPA']
    start_point = all_landmarks['RPA']
    end_point = all_landmarks['LPA']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Cz']], vertices[end_point]-vertices[all_landmarks['Cz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles([10,90], path, path_faces)
    
    vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)

    assert len(names) == len(el_position)
    for idx, name in enumerate(names):
        all_landmarks[name] = added_points[idx]
    ###################################################################
    # left transverse line through Oz aboveLPA Fpz
    start_point = all_landmarks['Oz']
    end_point = all_landmarks['Fpz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['aboveLPA']], vertices[end_point]-vertices[all_landmarks['aboveLPA']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles(np.linspace(0,100,10*subdivisions, endpoint = False)[1:], path, path_faces)

    vertices, faces, left_transverse = add_points(vertices, faces, el_position, el_faces)

    # left_transverse = np.arange(vertices.shape[0]-len(el_position), vertices.shape[0])
    left_transverse = np.concatenate([[all_landmarks['Oz']], left_transverse, [all_landmarks['Fpz']]])
    all_landmarks['T7'] = left_transverse[5*subdivisions]
    all_pos.append(left_transverse[1:-1])
    ##################################
    # right transverse line through Oz aboveRPA Fpz
    start_point = all_landmarks['Oz']
    end_point = all_landmarks['Fpz']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['aboveRPA']], vertices[end_point]-vertices[all_landmarks['aboveRPA']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
    el_position, el_faces = length_percentiles(np.linspace(0,100,10*subdivisions, endpoint = False)[1:], path, path_faces)

    vertices, faces, right_transverse = add_points(vertices, faces, el_position, el_faces)

    # right_transverse = np.arange(vertices.shape[0]-len(el_position), vertices.shape[0])
    right_transverse = np.concatenate([[all_landmarks['Oz']], right_transverse, [all_landmarks['Fpz']]])
    all_landmarks['T8'] = right_transverse[5*subdivisions]
    all_pos.append(right_transverse[1:-1])
    ###################################################################
    # add positions below Oz
    start_point = all_landmarks['T8']
    end_point = all_landmarks['T7']
    plane_normal = np.cross(vertices[start_point]-vertices[all_landmarks['Cz']], vertices[end_point]-vertices[all_landmarks['Cz']])
    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)

    # store 10% of the length of the previous (upper) path
    ten_percent = path_length(path)/10

    path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point, lower = True)

    # compute the corresponding percentiles in the lower path
    ten_percent = 100*ten_percent/path_length(path)

    el_position, el_faces = length_percentiles(ten_percent*np.linspace(1,0,subdivisions, endpoint = False), path, path_faces)
    vertices, faces, left_belowline = add_points(vertices, faces, el_position, el_faces)
    # left_belowline = np.arange(vertices.shape[0]-len(el_position), vertices.shape[0])
    # all_pos.append(left_belowline)

    el_position, el_faces = length_percentiles(100-ten_percent*np.linspace(1,0,subdivisions, endpoint = False), path, path_faces)
    vertices, faces, right_belowline = add_points(vertices, faces, el_position, el_faces)
    # right_belowline = np.arange(vertices.shape[0]-len(el_position), vertices.shape[0])
    # all_pos.append(right_belowline)
    ###################################################################
    # left transverse lines below Oz
    for i in range(subdivisions):
        start_point = sagittal_midline[i]
        end_point = sagittal_midline[-i-1]
        plane_normal = np.cross(vertices[start_point]-vertices[left_belowline[i]], vertices[end_point]-vertices[left_belowline[i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
        el_position, el_faces = length_percentiles(np.linspace(0,100,10*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
        # all_pos.append(np.arange(vertices.shape[0]-len(el_position), vertices.shape[0]))
        # vertices, faces = add_points(vertices, faces, el_position[:5*subdivisions-1], el_faces[:5*subdivisions-1])
        # all_pos.append(np.arange(vertices.shape[0]-len(el_position[:5*subdivisions-1]), vertices.shape[0]))
        # vertices, faces = add_points(vertices, faces, el_position[5*subdivisions:], el_faces[5*subdivisions:])
        # all_pos.append(np.arange(vertices.shape[0]-len(el_position[5*subdivisions:]), vertices.shape[0]))
    ###################################################################
    # right transverse lines below Oz
    for i in range(subdivisions):
        start_point = sagittal_midline[i]
        end_point = sagittal_midline[-i-1]
        plane_normal = np.cross(vertices[start_point]-vertices[right_belowline[i]], vertices[end_point]-vertices[right_belowline[i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=end_point)
        el_position, el_faces = length_percentiles(np.linspace(0,100,10*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
        # all_pos.append(np.arange(vertices.shape[0]-len(el_position), vertices.shape[0]))
        # vertices, faces = add_points(vertices, faces, el_position[:5*subdivisions-1], el_faces[:5*subdivisions-1])
        # all_pos.append(np.arange(vertices.shape[0]-len(el_position[:5*subdivisions-1]), vertices.shape[0]))
        # vertices, faces = add_points(vertices, faces, el_position[5*subdivisions:], el_faces[5*subdivisions:])
        # all_pos.append(np.arange(vertices.shape[0]-len(el_position[5*subdivisions:]), vertices.shape[0]))
    ###################################################################
    # short posterior coronal lines
    for i in range(subdivisions+1, 2*subdivisions):
        # right part
        start_point = right_transverse[i]
        end_point = left_transverse[i]
        plane_normal = np.cross(vertices[start_point]-vertices[sagittal_midline[i]], vertices[end_point]-vertices[sagittal_midline[i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=sagittal_midline[i])
        el_position, el_faces = length_percentiles(np.linspace(0,100,2*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
        
        # left part
        start_point = right_transverse[i]
        end_point = left_transverse[i]
        plane_normal = np.cross(vertices[start_point]-vertices[sagittal_midline[i]], vertices[end_point]-vertices[sagittal_midline[i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=sagittal_midline[i], end_point=end_point)
        el_position, el_faces = length_percentiles(np.linspace(0,100,2*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
    ###################################################################
    # short anterior coronal lines
    for i in range(subdivisions+1, 2*subdivisions):
        i+=1
        
        # right part
        start_point = right_transverse[-i]
        end_point = left_transverse[-i]
        plane_normal = np.cross(vertices[start_point]-vertices[sagittal_midline[-i]], vertices[end_point]-vertices[sagittal_midline[-i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=sagittal_midline[-i])
        el_position, el_faces = length_percentiles(np.linspace(0,100,2*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
        
        # left part
        start_point = right_transverse[-i]
        end_point = left_transverse[-i]
        plane_normal = np.cross(vertices[start_point]-vertices[sagittal_midline[-i]], vertices[end_point]-vertices[sagittal_midline[-i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=sagittal_midline[-i], end_point=end_point)
        el_position, el_faces = length_percentiles(np.linspace(0,100,2*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
    ###################################################################
    for i in range(2*subdivisions, 8*subdivisions+1):
        # right part
        start_point = right_transverse[i]
        end_point = left_transverse[i]
        plane_normal = np.cross(vertices[start_point]-vertices[sagittal_midline[i]], vertices[end_point]-vertices[sagittal_midline[i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=start_point, end_point=sagittal_midline[i])
        el_position, el_faces = length_percentiles(np.linspace(0,100,4*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
        
        # left part
        start_point = right_transverse[i]
        end_point = left_transverse[i]
        plane_normal = np.cross(vertices[start_point]-vertices[sagittal_midline[i]], vertices[end_point]-vertices[sagittal_midline[i]])
        path, path_faces = get_upper_path(vertices, faces, path_plane_normal=plane_normal, start_point=sagittal_midline[i], end_point=end_point)
        el_position, el_faces = length_percentiles(np.linspace(0,100,4*subdivisions, endpoint = False)[1:], path, path_faces)
        vertices, faces, added_points = add_points(vertices, faces, el_position, el_faces)
        all_pos.append(added_points)
    
    
    all_pos = np.concatenate(all_pos)

    if not return_indices:
        out = (vertices[all_pos],)
        for key, val in all_landmarks.items():
            all_landmarks[key] = vertices[val]
    else:
        out = (vertices, faces, all_pos)
    
    if return_landmarks:
        out += (all_landmarks,)
    
    return out



def create_random_montage(vertices, faces, fiducials,  min_dist = None, num_electrodes = None, sampling = 'poisson', generator = None, return_indices = False, return_landmarks = True):
    # creates a random montage on the input head
    # the electrodes are placed above the lowest line of electrodes in the 10-10 system, i.e. above the circle identified by [Oz T10 Nz T9 Oz]
    # min_dist: minimum distance between electrodes
    # num_electrodes: desired number of electrodes (approximate)

    assert sampling in ['poisson', 'uniform'], 'sampling must be either "poisson", for Poisson disk sampling, or "uniform", for uniform sampling'
    if sampling == 'uniform':
        assert isinstance(num_electrodes, int), 'You must specify the number of electrodes when sampling uniformly!'
    
    # compute 10-5 system
    vertices, faces, all_landmarks = create_standard_montage(vertices, faces, fiducials = fiducials, system = 'all', return_indices = True)


    # lowest line of electrodes, marks the lower bound on electrodes height
    belowline = ['Iz', 'I2h', 'I2', 'POO10', 'PO10', 'PPO10', 'P10', 'TPP10', 'TP10', 'TTP10', 'T10', 'FTT10', 'FT10', 'FFT10', 'F10', 'AFF10', 'AF10', 'AFp10', 'N2', 'N2h', 'Nz', 'N1h', 'N1', 'AFp9', 'AF9', 'AFF9', 'F9', 'FFT9', 'FT9', 'FTT9', 'T9', 'TTP9', 'TP9', 'TPP9', 'P9', 'PPO9', 'PO9', 'POO9', 'I1', 'I1h', 'Iz']
    belowline = list(map(lambda x: all_landmarks[x], belowline))


    # create a graph from the mesh to allow fast shortest path computations
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # edges without duplication
    edges = mesh.edges_unique

    to_be_removed = compute_path(belowline, vertices, faces)

    to_be_removed = np.unique(to_be_removed)


    # create the corresponding graph to compute connected component
    G = nx.Graph()
    for edge in edges:
        G.add_edge(*edge)

    for node in to_be_removed:
        G.remove_edges_from(list(zip([node]*len(G.adj[node]), G.neighbors(node))))

    # sorted connected components
    all_comp = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]

    if all_landmarks['Cz'] in all_comp[0]:
        to_be_removed = list(all_comp[1])
    elif all_landmarks['Cz'] in all_comp[1]:
        to_be_removed = list(all_comp[0])
    else:
        raise BaseException('something weird happened...')

    submesh = np.setdiff1d(np.arange(len(vertices)), to_be_removed)

    newverts, newfac, orig_faces = extract_submesh(vertices, faces, submesh, return_faces = True)

    print('Performing Sampling on the mesh...')
    if sampling == 'poisson':
        newverts, newfac, sampled_electrodes, sampled_faces = poisson_disk_sampling(newverts, newfac, min_dist=min_dist, num_points = num_electrodes, return_original_faces = True, generator = generator, remesh = False)
    elif sampling == 'uniform':
        newverts, newfac, sampled_electrodes, sampled_faces = uniform_sampling(newverts, newfac, num_points = num_electrodes, remesh = False, return_original_faces = True, generator = generator)
        
    # project sampled points on old mesh
    print('Projecting sampled points on original mesh...')
    sampled_faces = orig_faces[np.array(sampled_faces)]
    vertices, faces, all_pos = add_points(vertices, faces, newverts[sampled_electrodes], sampled_faces)

    if not return_indices:
        out = (vertices[all_pos],)
        for key, val in all_landmarks.items():
            all_landmarks[key] = vertices[val]
    else:
        out = (vertices, faces, all_pos)

    if return_landmarks:
        out += (all_landmarks,)

    return out
