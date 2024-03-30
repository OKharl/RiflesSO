# =================================================================================================
# RiflesSO_main.py: the main module calling an algorithm toolchain to evaluate the reflectance
# -------------------------------------------------------------------------------------------------
# The toolchain may include:
#     * initialization of the crystal and the band structure, 
#     * reflectance evaluation via the physical approximation selected,
#     * projection of the final states onto the finite k grid,
#     * export of the results using a chosen data format.
# Some methods are to be worked out/implemented in the future releases.
# -------------------------------------------------------------------------------------------------
# Created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Optional, List, Tuple, Set, Any, Union, Sequence, IO, Type
from numpy.typing import NDArray
from abc import ABC
import math
import numpy as np
from scipy.optimize import root_scalar
import yaml
from argparse import ArgumentParser
#import pytorch

from .lattice import CrystallineHalfSpace
from . import units
from .utils import timeit, normalize, nparr2str
from .elecstructure import ElectronicStructure, BlochState
from .boundarypot import BoundaryPotential, KinkPotential
from .projector import KgridProjector

class ReflectionTask:
    '''A class encapsulating calculation of a specific reflection property using a certain method/approximation,
    e.g., of a reflectance matrix for a wall with a specific normal vector and boundary potential profile.
    '''
    engine:                 'RiflesSO'
    lattice:                CrystallineHalfSpace
    electronic_structure:   Type[ElectronicStructure]
    boundary_potential:     Type[BoundaryPotential]
    reflection_solver:      Type['ReflectionSolver']
    kgrid_projector:        Type['KgridProjector']

    calctype:               str         # Calculation type: 'single_WF_reflection', 'all_WF_reflection'
    k_frac:                 NDArray[np.float]  # Fractional components of incident wave's \vec{k} vector
    in_state:               BlochState  # A state to calculate reflection of for calctype = 'single_WF_reflection'
    
    # The result(s) of the calculation in the format [(in_state, (A_1, out_state1), (A_2, out_state2), ...}), ...]
    calc_result:            List[Tuple[BlochState, Set[Tuple[complex, BlochState]]]] 

    def __init__(self, engine: 'RiflesSO', calctype: str, *, 
                 k_frac: Optional[NDArray[np.float]] = None, 
                 in_state: Optional[BlochState] = None):
        self.engine = engine
        self.calctype = calctype
        self.lattice = engine.lattice
        self.electronic_structure = engine.electronic_structure
        self.boundary_potential = engine.boundary_potential
        self.reflection_solver = engine.reflection_solver
        self.kgrid_projector = engine.kgrid_projector
        
        self.k_frac = k_frac
        self.in_state = in_state
        self.calc_result = None

    def run(self) -> Any:
        '''Performs the configured reflectance calculation. 
        Returns the result once done; its format may depend on the particular calculation, 
        but generally includes the reflection amplitudes.
        '''
        if self.calctype == 'single_WF_reflection':
            self.in_state.k_frac = self.k_frac   # For safety
            self.calc_result = [(self.in_state, self.reflection_solver.reflect_wavefunction(self.in_state))]
        elif self.calctype == 'all_WF_reflection':
            in_states = self.electronic_structure.electronic_states_BZ(self.k_frac)
            self.calc_result = [(s, self.reflection_solver.reflect_wavefunction(s)) for s in in_states]
        else:
            raise ValueError(f'In ReflectionTask.run(): cannot recognize the calculation type "{self.calctype}"')
        return self.calc_result

    def result(self) -> dict:
        "Returns the result of the calculation once it is complete"
        return {'reflection_coeffs': self.calc_result}
    
    

from .reflsolver import ReflectionSolver

class RiflesSO:
    '''The main entry point of the reflectance solver toolchain. Can be initialized with 
    command-line arguments, a YAML script, or step by step manually. Typically, the chain contains
    initialization of the electronic structure, calculation of the reflectance, and projection
    of the latter onto a given finite momentum grid.
    '''
    
    arg_parser: ArgumentParser

    lattice:                CrystallineHalfSpace
    electronic_structure:   ElectronicStructure
    boundary_potential:     BoundaryPotential
    reflection_solver:      'ReflectionSolver'
    kgrid_projector:        'KgridProjector'

    log_stream:             IO
    log_filename:           str

    def __init__(self, *, 
                 from_command_line: Optional[str] = None,
                 from_dict: Optional[dict] = None,
                 from_yaml: Optional[Union[str, yaml.YAMLObject, dict]] = None
                ):
        self.create_argument_parser()
        self.log_filename = 'console'
        self.log_stream = None
        self.lattice = CrystallineHalfSpace(lattice_vectors = np.identity(3, dtype=float) * units.Angstrom,
                                            boundary_plane_indices = [0, 0, 1]
                                           )
        self.electronic_structure = None
        self.boundary_potential = KinkPotential(shape='kink.tanh', V_0 = 10 * units.eV, V_width = 10 * units.Angstrom)
        self.reflection_solver = None
        self.kgrid_projector = None

        if from_dict:
            self.initialize_from_dictionary(from_dict)
        elif from_yaml:
            if isinstance(from_yaml, str):
                with open(from_yaml, 'rt') as f:
                    yaml_dict = yaml.safe_load(f)
            elif isinstance(from_yaml, yaml.YAMLObject):
                yaml_dict = dict(from_yaml)  # NOT SURE...
            elif isinstance(from_yaml, dict):
                yaml_dict = from_yaml
            self.initialize_from_dictionary(yaml_dict['riflesso'])
        elif from_command_line:
            self.initialize_from_command_line(from_command_line)
        else:
            self.initialize_from_command_line('-o riflesso')

    def __del__(self):
        if self.log_filename != 'console' and self.log_stream is not None:
            self.log_stream.close()

    def run(self):
        pass

    def initialize_from_command_line(self, command_line: Optional[str] = None):
        if command_line:
            self.arg_parser.parse(command_line)
            
    
    def log(self, msg: str, *args, **kwargs):
        if self.log_filename == 'console':
            print(msg, *args, **kwargs)
        elif self.log_filename and self.log_stream is not None:
            print(msg, file=self.log_stream, *args, **kwargs)

    def initialize_from_dictionary(self, d: dict):
        if 'lattice' in d:
            d_lattice = dict(d['lattice'])
            if 'boundary-normal.Cartesian' not in d_lattice and 'Miller_indices' not in d_lattice:
                # Add a dummy normal vector, otherwise a CrystallineHalfSpace object may not initialize correctly
                d_lattice['Miller_indices'] = [0, 0, 1]
            self.lattice = CrystallineHalfSpace()
            self.lattice.initialize_from_dictionary(d_lattice)
            self.log('Crystal lattice initialized with the following primitive vectors [in Angstroms]:\n\t' + 
                     '\n\t'.join(['a', 'b', 'c'][i] + ' = ' 
                                 + nparr2str(self.lattice.lattice_vectors[:,i] / units.Angstrom) 
                                 for i in range(self.lattice.crystal_dimension())
                                ),
                     sep=''
                    )
        if 'electronic-structure' in d:
            self.electronic_structure = ElectronicStructure.from_dictionary(d['electronic-structure'])
            self.electronic_structure.lattice = self.lattice
            self.log(f'Electronic structure initialized: {self.electronic_structure}')
        if 'boundary-potential' in d:
            self.boundary_potential = BoundaryPotential.from_dictionary(d['boundary-potential'])
        elif self.boundary_potential is None: # Create a default one
            self.boundary_potential = KinkPotential(shape='tanh', V_0 = 10 * units.eV, V_width = 10 * units.Angstrom)
        self.log(f'Boundary potential fixed: {self.boundary_potential}')
        if 'solver' in d:
            self.reflection_solver = ReflectionSolver.from_dictionary(d['solver'])
            self.log(f'Reflection solver initialized: {self.reflection_solver}')
        # TODO: parse other keywords


    def create_argument_parser(self):
        "Create a parser object for potential scanning of the command line passed to RiflesSO"
        self.arg_parser = ArgumentParser(prog='RiflesSO', 
                                         description='ab initio reflectance solver for crystalline materials')
        self.arg_parser.add_argument('-i', '--input-yaml', type=str, 
                                     help='Input YAML filename describing the tasks to be done', required=False)
        self.arg_parser.add_argument('-o', '--output-stem', type=str, 
                                     help='Stem for output files, e.g., those with the calculated reflectances', required=False)



# -------------------------------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    jdftx_dir = '../../jdftx_calculations/'
    ref = RiflesSO() #Reflector()
    ref.lattice = CrystallineHalfSpace(
        lattice_vectors = (5.34145 * units.BohrRadius) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        boundary_plane_indices = [1, 0, 0]
    )
    
    tests2do = [ 'reflection.0th_adiabatic[material=graphene]' ]
    # [ 'bandstruct.state_propagation[material=graphene]', 'bandstruct.DiracPoint[material=graphene]'
    #   'bandstruct.wannierization[material=graphene]',
    #   'reflection.0th_adiabatic[material=graphene]' ] 
    for test in tests2do:
        material = 'GaAs'
        substr = 'material='
        if substr in test:
            pos_start = test.find(substr) + len(substr)
            pos_end = pos_start
            while pos_end < len(test) and test[pos_end].isalnum():
                pos_end += 1
            material = test[pos_start : pos_end]
        calcdir = jdftx_dir + material + '/'
        outdir = '../test/' + material + '/'
        ref.load_electronic_structure('jdftx.wannierized', calcdir + 'scf_fine', calcdir + 'wannier')
        # The tests themselves
        if 'bandstruct.wannierization' in test:
            print('-' * 80 + f'\nTest: band structure along a path in k-space [{material}]')
            kpoints = np.loadtxt(calcdir + 'bandstruct_path.kpoints', skiprows=2, usecols=(1,2,3)).astype(float)
            #kpoints = kpoints[0::5]
            energies = np.array([[state[2] for state in ref.electronic_states_BZ(k)] for k in kpoints])
            np.savetxt(outdir + 'band_energies.dat', np.column_stack([kpoints, energies / units.eV]), 
                       header='k_x[frac] k_y[frac] k_z[frac] E_1[eV] ... E_n[eV]', fmt='%.6f')
        elif 'bandstruct.state_propagation' in test:
            print('-' * 80 + f'\nTest: Adiabatic propagation of states [{material}]')
            kpoints = np.loadtxt(calcdir + 'bandstruct_path.kpoints', skiprows=2, usecols=(1,2,3)).astype(float)[::2]
            states_k = ref.electronic_states_BZ(kpoints[0])
            nbands = len(states_k)
            propagated_energies = [[state[2] for state in states_k]]
            for i in range(1, len(kpoints)):
                propagated_states = [ref.propagate_band_in_BZ(states_k[n], kpoints[i]) for n in range(nbands)]
                propagated_energies.append([state[2] for state in propagated_states])
                states_k = propagated_states
                print(f'Energies propagated to k-point #{i + 1} out of {len(kpoints)}!')
            np.savetxt(outdir + 'propagated_energies.dat', np.array(propagated_energies) / units.eV, 
                       header='E_1[eV] ... E_n[eV]', fmt='%.6f')
        elif 'reflection.0th_adiabatic' in test:
            print('-' * 80 + f'\nTest: Reflection in the 0th adiabatic approximation [{material}]')
            kpoints = np.loadtxt(calcdir + 'bandstruct_path.kpoints', skiprows=2, usecols=(1,2,3)).astype(float)
            k_in = kpoints[10]
            n_Cartesian = normalize(ref.lattice.boundary_normal())
            v_n = 0.0
            n_in = 0
            while True: # find a well-defined incident state
                k_Cartesian = ref.lattice.fractional_coords_to_reciprocal_vector(k_in)
                state_in = ref.electronic_states_BZ(k_in)[n_in]
                v_n = state_in[4].dot(n_Cartesian)
                if v_n <= 0:
                    n_in += 1
                else:
                    break
            print(f'For k_frac = {nparr2str(k_in, 4)} [frac] = {nparr2str(k_Cartesian * units.Angstrom)} Ao^(-1),',
                f'\nreflection of state #{state_in[1]} with E = {state_in[2] / units.eV : .3f} eV',
                f'and v_n = {nparr2str(state_in[4].dot(n_Cartesian) / (1e8 * units.cm / units.sec))} [1e8 cm/sec]',
                f'\noff a plane with the normal n = {nparr2str(n_Cartesian)} gives:')
            states_out = ref.reflect_wavefunction(state_in)
            for prob, s in states_out:
                print(f'   * probability = {prob},',
                    f'k_frac = {nparr2str(s[0])}, band = {s[1]}, E = {s[2] / units.eV : .3f} eV,',
                    '\n   * chi = ' + np.array2string(s[3], precision=1, floatmode='fixed', max_line_width=None, prefix='   * chi = ')
                    )
            
            if material == 'graphene':
                k_Kpoint_frac = np.array([1/3, 1/3, 0])
                k_Kpoint_cart = ref.lattice.fractional_coords_to_reciprocal_vector(k_Kpoint_frac)
                reflection_data = []
                for phi in np.linspace(0, 2 * np.pi, 100):
                    k_cart = k_Kpoint_cart + 0.025 * np.linalg.norm(k_Kpoint_cart) * np.array([math.cos(phi), math.sin(phi), 0.0])
                    k_frac = ref.lattice.reciprocal_vector_to_fractional_coords(k_cart)
                    for state_in in ref.electronic_states_BZ(k_frac):
                        if state_in[4].dot(n_Cartesian) <= 0.0:  # Not an incident state
                            continue
                        if abs(state_in[2] + 4.23 * units.eV) > 2.5 * units.eV:  # Too far from Fermi surface
                            continue
                        print(f'Initial state with k_frac = {nparr2str(k_frac)}, band = {state_in[1]}, E = {state_in[2] / units.eV: .5f} eV reflects to:')
                        states_out = ref.reflect_wavefunction(state_in)
                        for prob, s in states_out:
                            print(f'   * [P = {prob: .3f}], k_frac = {nparr2str(s[0])}, band = {s[1]}, E = {s[2] / units.eV : .5f} eV')
                            kprime_cart = ref.lattice.fractional_coords_to_reciprocal_vector(np.mod(s[0], 1.0))
                            reflection_data.append([*(k_cart * units.Angstrom), state_in[1], state_in[2] / units.eV,
                                                    prob,
                                                    *(kprime_cart * units.Angstrom), s[1], s[2] / units.eV
                                                   ])
                np.savetxt(outdir + 'reflection_data.dat', reflection_data, 
                       header='k_x[Ao-1] k_y[Ao-1] k_z[Ao-1] n E[eV] P_refl kprime_x[Ao-1] kprime_y[Ao-1] kprime_z[Ao-1] nprime Eprime[eV]', 
                       fmt='%.3f %.3f %.3f %d %.3f %.3f %.3f %.3f %.3f %d %.3f'
                )
        elif 'bandstruct.DiracPoint[material=graphene]' in test:
            print('-' * 80 + f'\nTest: Energy bands near the Dirac points [{material}]')
            k_Kpoint_frac = np.array([1/3, 1/3, 0])
            bands_data = []
            kpts_frac = [k_Kpoint_frac + dk * np.array([math.cos(phi), math.sin(phi), 0.0]) 
                         for dk in np.linspace(0, 0.1, 10) 
                         for phi in np.linspace(0, 2 * np.pi, 100)
                        ]
            for k_frac in kpts_frac:
                for s in ref.electronic_states_BZ(k_frac):
                    if abs(s[2] + 4.23 * units.eV) > 2.5 * units.eV:  # Too far from Fermi surface
                        continue
                    k_cart = ref.lattice.fractional_coords_to_reciprocal_vector(s[0])
                    print(f'Found state with k_frac = {nparr2str(k_frac)}, band = {s[1]}, E = {s[2] / units.eV: .3f} eV')
                    bands_data.append([*(k_cart * units.Angstrom), s[1], s[2] / units.eV])
            np.savetxt(outdir + 'bands_Kvalley_data.dat', bands_data, 
                       header='k_x[Ao-1] k_y[Ao-1] k_z[Ao-1] n E[eV]', 
                       fmt='%.6f %.6f %.1f %d %.6f'
            )

        
    




