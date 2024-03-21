# =================================================================================================
# ReflSolver_adiabatic: reflection solver for an adiabatically smooth boundary potential V(z)
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import List, Tuple, Set, Optional
import math
import numpy as np

from .riflesso_main import ReflectionTask
from .reflsolver import ReflectionSolver
from .elecstructure import BlochState
from . import units

class AdiabaticReflectionSolver(ReflectionSolver):
    '''A simple reflection solver formally corresponding to the 0th adiabatic approximation
    for the boundary potential. Any incident state with energy E is "parallel transported" 
    across the k space until it pops us at the energy surface again as a reflected state;
    thereby, a single reflected state is found per each incident one.
    '''

    z_cutoff_lattice_units: float           # The cutoff distance from the boundary in lattice units (should be >> 1)
    dq: float                               # Default momentum shift per step
    E_tolerance: float                      # Energy tolerance for finding the final state with E_final = E_initial

    def __init__(self, *, params: Optional[dict] = None, task: Optional[ReflectionTask] = None):
        super().__init__(params, task)
        self.z_cutoff_lattice_units = 25.0
        self.dq = 0.001
        self.E_tolerance = 0.0001 * units.eV

    def initialize_params(self, params: Optional[dict] = None):
        '''Initialize the solver's parameters with a dictionary (say, from a parsed configuration file).
        An empty dictionary or None value should install the defaults. 
        Specific parameter names and syntax depend on the particular derived solver class's implementation.
        '''
        p = params['solver-params'] if params and 'solver-params' in params else {}
        self.z_cutoff_lattice_units = float(p.get('cutoff-distance-lattice-units', 25.0))
        self.dq = p.get('k-step', 0.001)
        self.E_tolerance = float(p.get('final-E-tolerance', 0.0001)) * units.eV


    def reflect_wavefunction(self, incident_state: BlochState) -> Set[Tuple[complex, BlochState]]:
        '''For an pure incident_state = (\\vec{k}, n, E, u, v), find a (single) reflected state s
        and return it together with its amplitude, {(A, s)}. Note that in the adiabatic approximation,
        |A| = 1, so that the probability P = 1.
        '''
        _implementation_version_ = 0
        t = self.task # Reflection task containing all the parameters
        # The below projection of \vec{k_frac} will change
        e_q = np.array(t.lattice.boundary_potential_kdirection(), dtype=float)
        #dk_over_dq = t.lattice.fractional_coords_to_reciprocal_vector(e_q)
        lattice_scale = np.mean([np.linalg.norm(v) for v in t.lattice.lattice_vectors.T])
        z_cutoff = self.z_cutoff_lattice_units * lattice_scale  # Let's assume the bulk starts here rather than at z → -∞
        z_in = -z_cutoff 
        k_in, E_in = incident_state.k_frac, incident_state.E # + self.boundary_potential_value(z_in)
        n_Cartesian = t.lattice.boundary_normal()
        z, q, phase = z_in, 0.0, 0.0       # Further, we will work with k = k_in + q * e_q
        state = incident_state
        while z > -1.01 * z_cutoff:
            dq = -self.dq            # To be chosen in a smarter, adaptive way in what follows
            propagated_state = t.electronic_structure.propagate_band_in_BZ(state, k_in + (q + dq) * e_q)
            z_next = t.boundary_potential.potential_inv_function(E_in - propagated_state.E)
            phase += np.dot(k_in + (q + 0.5 * dq) * e_q, (z_next - z) * n_Cartesian)
            state, z, q = propagated_state, z_next, q + dq
        # [UNFINISHED] Correct q to get the energy equal to E_in
        # We need E_n(q_corr) = E_in, but are having E(q) instead, thus, q_corr ~ q + dq, where
        # E_in - E(q) = dq * dE_n/dq = dq * hbar (v_n dk/dq) = dq * hbar (v_n * frac2cart(e_q))
        #dq_corr = (E_in - state[2]) / (units.hbar * propagated_state[4].dot(dk_over_dq))
        #if abs(dq_corr) < abs(dq):
        #    state = self.propagate_band_in_BZ(state, k_in + (q + dq_corr) * e_q)
        state = t.electronic_structure.propagate_band_in_energy(state, E_in, e_q, 
                                                                options={'E_tolerance': self.E_tolerance, 
                                                                         'max_iterations': 50}
                                                               )
        
        # In the adiabatic regime, there is only one final state with P = 1
        return {(math.exp(1.0j * phase), state)}
