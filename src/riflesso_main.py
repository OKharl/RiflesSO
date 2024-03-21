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

from typing import Optional, List, Tuple, Set, Any
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
from .elecstructure import ElectronicStructure, WannierElectronicStructure, BlochState
from .boundarypot import BoundaryPotential
from .reflsolver import ReflectionSolver
from .projector import KgridProjector


class ReflectionTask:
    '''A class encapsulating calculation of a specific reflection property using a certain method/approximation,
    e.g., of a reflectance matrix for a wall with a specific normal vector and boundary potential profile.
    '''
    engine:                 'RiflesSO'
    lattice:                CrystallineHalfSpace
    electronic_structure:   ElectronicStructure
    boundary_potential:     BoundaryPotential
    reflection_solver:      'ReflectionSolver'
    kgrid_projector:        'KgridProjector'

    calctype:               str         # Calculation type: 'single_WF_reflection', 'all_WF_reflection'
    k_frac:                 np.ndarray  # Fractional components of incident wave's \vec{k} vector
    in_state:               BlochState  # A state to calculate reflection of for calctype = 'single_WF_reflection'
    
    # The result(s) of the calculation in the format [(in_state, (A_1, out_state1), (A_2, out_state2), ...}), ...]
    calc_result:            List[Tuple[BlochState, Set[Tuple[complex, BlochState]]]] 

    def __init__(self, engine: 'RiflesSO', calctype: str, *, 
                 k_frac: np.ndarray, in_state: BlochState):
        pass

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


class RiflesSO:
    '''The main entry point of the reflectance solver toolchain. Can be initialized with 
    command-line arguments, a YAML script, or step by step manually.
    '''
    
    arg_parser: ArgumentParser

    def __init__(self, *, 
                 from_command_line: Optional[str] = None,
                 from_dict: Optional[dict] = None,
                 from_yaml: Optional[Union[str, yaml.YAMLObject, dict]] = None
                ):
        self.create_argument_parser()
        if from_dict:
            self.initialize_from_dictionary(from_dict)
        elif from_yaml:
            if from_yaml is str:
                with open(from_yaml, 'rt') as f:
                    yaml_dict = yaml.safe_load(f)
            elif from_yaml is yaml.YAMLObject:
                yaml_dict = from_yaml  # NOT SURE...
            elif from_yaml is dict:
                yaml_dict = from_yaml
            self.initialize_from_dictionary(yaml_dict)
        elif from_command_line:
            self.initialize_from_command_line(from_command_line)
        else:
            self.initialize_from_command_line('-o riflesso')

    def run(self):
        pass

    def initialize_from_command_line(self, command_line: Optional[str] = None):
        if command_line:
            self.arg_parser.parse(command_line)
            # TODO: process the arguments
            # ...

    def initialize_from_dictionary(self, d: dict):
        # TODO: process the dict fields
        # ...
        pass

    def create_argument_parser(self):
        "Create a parser object for potential scanning of the command line passed to RiflesSO"
        self.arg_parser = ArgumentParser(prog='RiflesSO', 
                                         description='ab initio reflectance solver for crystalline materials')
        self.arg_parser.add_argument('-i', '--input-yaml', type='str', 
                                     help='Input YAML filename describing the tasks to be done', required=False)
        self.arg_parser.add_argument('-o', '--output-stem', type='str', 
                                     help='Stem for output files, e.g., those with the calculated reflectances', required=False)





class Reflector:
    '''
    The main reflectance solver class implementing various approaches/approximations and band structure import.
    '''
    
    lattice: CrystallineHalfSpace        #: stores the boundary and lattice vectors
    boundary_potential: Tuple[str, ...]  #: boundary potential type and its params, tanh supported so far

    def __init__(self, *, 
                 lattice_vectors: Sequence[Sequence[float]] = np.identity(3)
                 ):
        self.lattice = CrystallineHalfSpace(lattice_vectors=lattice_vectors)
        self.boundary_potential = ('tanh', 10.0 * units.eV, 25.0 * units.Angstrom)
    

    def reflect_wavefunction(self, incident_state: Tuple) -> Tuple[float, tuple]:
        '''
        For an incident pure state incident_state = (\\vec{k}, n, E, chi, v), 
        find the reflected states s_a = (\\vec{k}_a, n_a, E_a, chi_a, v_a) and the
        corresponding reflection probabilities P_a. Returns a set {(P_a, s_a)}.
        '''
        # Version 1: 0th adiabatic approximation (to be implemented in a separate class)
        e_q = np.array(self.lattice.boundary_potential_kdirection(), dtype=float) # this projection of \vec{k} will change
        dk_over_dq = self.lattice.fractional_coords_to_reciprocal_vector(e_q)
        lattice_scale = np.mean([np.linalg.norm(v) for v in self.lattice.lattice_vectors.T])
        z_cutoff = 25.0 * lattice_scale  # Let's assume the bulk starts here rather than at z → -∞
        z_in = -z_cutoff 
        k_in, E_in = incident_state[0], incident_state[2] # + self.boundary_potential_value(z_in)
        n_Cartesian = self.lattice.boundary_normal()
        z, q, phase = z_in, 0.0, 0.0       # Further, we will work with k = k_in + q * e_q
        state = incident_state
        while z > -1.01 * z_cutoff:
            dq = -0.001            # To be chosen in a smarter, adaptive way in what follows
            propagated_state = self.propagate_band_in_BZ(state, k_in + (q + dq) * e_q)
            z_next = self.boundary_potential_inv(E_in - propagated_state[2])
            phase += np.dot(k_in + (q + 0.5 * dq) * e_q, (z_next - z) * n_Cartesian)
            state, z, q = propagated_state, z_next, q + dq
        # [UNFINISHED] Correct q to get the energy equal to E_in
        # We need E_n(q_corr) = E_in, but are having E(q) instead, thus, q_corr ~ q + dq, where
        # E_in - E(q) = dq * dE_n/dq = dq * hbar (v_n dk/dq) = dq * hbar (v_n * frac2cart(e_q))
        #dq_corr = (E_in - state[2]) / (units.hbar * propagated_state[4].dot(dk_over_dq))
        #if abs(dq_corr) < abs(dq):
        #    state = self.propagate_band_in_BZ(state, k_in + (q + dq_corr) * e_q)
        state = self.propagate_band_in_energy(state, E_in, e_q)
        
        # In fact, in the adiabatic regime, there is only one final state, and the phase is unimportant
        return [(1.0, state)]



if __name__ == '__main__':
    jdftx_dir = '../../jdftx_calculations/'
    ref = Reflector()
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

        
    




