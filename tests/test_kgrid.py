# =================================================================================================
# Test_Kgrid.py: unit tests for Kgrid class (projector.py)
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

import numpy as np
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.lattice import *
from src.projector import *
from src.utils import *
from src.units import *

separ = '-' * 100

if __name__ == '__main__':
    # -------------------------------------------------------------------------------
    # Test 1: Create a lattice and a k grid based on it
    # -------------------------------------------------------------------------------
    print(separ, 'Test 1: Create a lattice and a k grid based on it', separ, sep='\n')
    with open('tests/main_toolchain/reflect_graphene.yaml', 'rt') as f:
        yaml = yaml.load(f, yaml.SafeLoader)
    yaml_lattice = yaml['riflesso']['crystal-lattice']
    print(f'Creating crystal lattice from {yaml_lattice}...')
    lattice = CrystalLattice()
    lattice.initialize_from_dictionary(yaml_lattice)
    print('Crystal lattice created with the following lattice vectors:\n\t',
          *('\n\t'.join(['a', 'b', 'c'][i] + ' = ' + nparr2str(lattice.lattice_vectors[:,i] / Angstrom) for i in range(3))),
          sep=''
         ) 
    grid = Kgrid(lattice=lattice, type='MP', npoints=[5, 4, 1])
    print(f'k grid created: {grid}, with a total of {grid.size()} points in it')

    # -------------------------------------------------------------------------------
    # Test 2: grid as an iterable
    # -------------------------------------------------------------------------------
    print(separ, 'Test 2: grid as an iterable', separ, sep='\n')
    print('Enumerating the fractional coordinates of the grid points via kpoints():')
    for idx, k in enumerate(grid.kpoints(Cartesian=False)):
        print(f'k-point #{idx + 1}: k_frac = {nparr2str(k)}')
    print('\nEnumerating the Cartesian coordinates of the grid points via __iter__():')
    for idx, k in enumerate(grid):
        print(f'k-point #{idx + 1}: k_frac = {nparr2str(k)}')
    print('\nChecking consistency of scalar and vector indices:')
    for idx in range(grid.size()):
        idx_vec = grid.point_index_scalar2vector(idx)
        idx_vec_scalar = grid.point_index_vector2scalar(idx_vec)
        print(f'Scalar index {idx} -> vector index {list(idx_vec)} -> scalar index {idx_vec_scalar}')

    # -------------------------------------------------------------------------------
    # Test 3: snapping to grid, etc.
    # -------------------------------------------------------------------------------
    print(separ, 'Test 3: snapping to grid, etc.', separ, sep='\n')
    for test in range(5):
        k_frac = np.random.random(3)
        k_frac[2] = 0.0
        idx = grid.closest_point_index(k_frac, Cartesian=False, scalar_index=True)
        print(f'k_frac = {nparr2str(k_frac)} is rounded to grid point #{idx}, ' + 
              f'i.e., to k_frac = {nparr2str(grid.grid_point_fractional(idx))}')
