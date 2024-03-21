# =================================================================================================
# ReflSolver.py: definition of the base class for reflectance solvers
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Optional, Sequence, List, Tuple, Union, Set
from abc import ABC
import numpy as np
from . import units
from .riflesso_main import ReflectionTask
from .elecstructure import BlochState


class ReflectionSolver(ABC):
    task: ReflectionTask

    def __init__(self, *, params: Optional[dict] = None, task: Optional[ReflectionTask] = None):
        '''Initialize the solver's parameters with a dictionary (say, from a parsed configuration file)
        and, optionally, associate it with a reflection task.
        '''
        self.task = task
        self.initialize_params(params)

    def initialize_params(params: Optional[dict] = None):
        '''Initialize the solver's parameters with a dictionary (say, from a parsed configuration file).
        An empty dictionary or None value should install the defaults. 
        Specific parameter names and syntax depend on the particular derived solver class's implementation.
        '''
        pass

    def reflect_density_matrix(self, k_fractional: np.array, rho_k: np.array) -> Set[Tuple[float, BlochState]]:
        '''For an incident wave with momentum \\vec{k}, described by the density matrix rho_k, 
        find the reflected states s_a = (\\vec{k}_a, n_a, E_a, u_a, v_a) with probabilities P_a.
        Returns a set {(P_a, s_a)}. Note that the reflected momenta do not have to lie 
        on the input k-grid.
        '''
        # The incident states and the outward normal to the boundary
        states_k = self.task.electronic_structure.electronic_states_BZ(k_fractional, 'eigenstates')
        normal_vec = self.task.lattice.boundary_normal()
        # Reflect each incident-state wavefunction and collect all of reflected parts
        reflected_states = []
        for s in states_k:
            prob_s = np.vdot(s.u, rho_k.dot(s.u)).real()     # Probability of the incident state s...
            influx_s = prob_s * np.dot(s.v, normal_vec)      # ...and its normal probability flux current
            if influx_s <= 0:   # Ignore these, "wrong/already reflected states"
                continue
            for refl_prob, refl_state in self.reflect_wavefunction(s):
                reflected_states.append([prob_s * refl_prob, refl_state])
        return set(reflected_states)

    def reflect_wavefunction(self, incident_state: BlochState) -> Set[Tuple[complex, BlochState]]:
        '''
        For a pure incident_state = (\\vec{k}, n, E, u, v), 
        find the reflected states s_a = (\\vec{k}_a, n_a, E_a, u_a, v_a) and the
        corresponding reflection probabilities P_a. Returns a set {(P_a, s_a)}.
        '''
        pass
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
