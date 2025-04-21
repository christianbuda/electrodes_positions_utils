from setuptools import setup, find_packages

setup(
    name='electrodes_positions',
    version='1.0',
    url='https://github.com/christianbuda/electrodes_positions_utils',
    author='Christian Buda',
    author_email='chrichri975@gmail.com',
    packages=find_packages(),
    install_requires = [
        'numpy',
        'scipy',
        'trimesh',
        'networkx',
        'pyvista',
        'tqdm',
        'autograd[scipy]',
        'mesh_poisson_disk_sampling'
    ]
)