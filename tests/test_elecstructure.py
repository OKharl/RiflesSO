# =================================================================================================
# Test_ElecStructure.py: unit tests for ElecStructure.py
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

from src import elecstructure as es
from src import lattice as lat
from src.utils import *
from src.units import *

separ = '-' * 100

if __name__ == '__main__':
    # -------------------------------------------------------------------------------
    # Test 1: Import electronic structure of graphene and test its basic props
    # -------------------------------------------------------------------------------
    print(separ, 'Test 1: Import electronic structure of graphene and test its basic props', separ, sep='\n')
    with open('tests/main_toolchain/reflect_graphene.yaml', 'rt') as f:
        yaml = yaml.load(f, yaml.SafeLoader)
    yaml_estruc = yaml['riflesso']['electronic-structure']
    yaml_lattice = yaml['riflesso']['crystal-lattice']
    print(f'Creating crystal lattice from {yaml_lattice}...')
    lattice = lat.CrystalLattice()
    lattice.initialize_from_dictionary(yaml_lattice)
    print('Crystal lattice created with the following lattice vectors:\n\t',
          *('\n\t'.join(['a', 'b', 'c'][i] + ' = ' + nparr2str(lattice.lattice_vectors[:,i] / Angstrom) for i in range(3))),
          sep=''
         ) 
    struc = es.ElectronicStructure.from_dictionary(yaml_estruc)
    print(f'Electronic structure imported: {struc}')

    print(f'Creating electronic structure from {yaml_estruc}...')
    struc = es.ElectronicStructure.from_dictionary(yaml_estruc)
    struc.lattice = lattice
    print(f'Electronic structure imported: {struc}')
    k_frac = np.array([0.33, 0.7, 0.0])
    k_states = struc.electronic_states_BZ(k_frac, return_value='eigenstates')
    nstates = len(k_states)
    print(f'At k = {nparr2str(k_frac)}, the {nstates} Bloch states are: ')
    for i, s in enumerate(k_states):
        print(f'Band {s.n}: E = {s.E / eV: .3f} eV, u = {nparr2str(s.u, prec=2)}')
    u_products = np.array([[np.vdot(k_states[i].u, k_states[j].u) for j in range(nstates)] for i in range(nstates)])
    print(f'Orthogonality test: |<u_i|u_j> - delta_ij| = {np.linalg.norm(u_products - np.identity(nstates))}')

    # -------------------------------------------------------------------------------
    # Test 2: Plot dispersion near K point
    # -------------------------------------------------------------------------------
    print(separ, 'Test 2: Plot dispersion near K point', separ, sep='\n')
    k_Kpoint = np.array([1/3, 1/3, 0])
    dk = 0.01
    E_shift = 4.23 * eV
    states_Kpoint = [s for s in struc.electronic_states_BZ(k_Kpoint) if abs(s.E + E_shift) < 0.5 * eV]
    bands = [s.n for s in states_Kpoint]
    print(f'{len(bands)} states found at the K point with small energies: bands = {bands}')
    kx_values = np.linspace(k_Kpoint[0] - dk, k_Kpoint[0] + dk, 25)
    ky_values = np.linspace(k_Kpoint[1] - dk, k_Kpoint[1] + dk, 25)
    kx_mesh, ky_mesh = np.meshgrid(kx_values, ky_values)
    E_mesh = np.zeros(shape=kx_mesh.shape + (len(bands), ))
    for i in range(kx_mesh.shape[0]):
        for j in range(kx_mesh.shape[1]):
            k_states = struc.electronic_states_BZ(k_Kpoint + np.array([kx_mesh[i,j], ky_mesh[i,j], 0]))
            E_selected = []
            for b in bands:
                for s in k_states:
                    if s.n == b:
                        E_selected.append(s.E + E_shift)
                        break
            E_mesh[i,j] = E_selected
    E_mesh = np.array(E_mesh)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for b in range(len(bands)):
        ax.plot_surface(kx_mesh - k_Kpoint[0], ky_mesh - k_Kpoint[1], E_mesh[:,:,b] / eV)
    ax.set_xlabel('$\Delta k_x$ [frac.]')
    ax.set_ylabel('$\Delta k_y$ [frac.]')
    ax.set_zlabel('$E_n(\\mathbf{k})$ [eV]')
    ax.view_init(elev=10, azim=40)
    #plt.show()
    png_filename = 'tests/main_toolchain/test_elecstructure.Kpoint_disp.png'
    plt.savefig(png_filename)
    print(f'Dispersion plot saved in {png_filename}')
