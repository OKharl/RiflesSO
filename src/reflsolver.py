# =================================================================================================
# ReflSolver.py: definition of the base class for reflectance solvers
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Optional, Tuple, Set, Type
from abc import ABC
import numpy as np

from .riflesso_main import ReflectionTask
from .elecstructure import BlochState

# Derived reflectance solver classes
#from .reflsolver_adiabatic import AdiabaticReflectionSolver

class ReflectionSolver(ABC):
    # Derived solver classes supported by from_dictionary() method
    solver_types = { 'adiabatic': 'AdiabaticReflectionSolver' }

    task: 'ReflectionTask'

    def __init__(self, *, params: Optional[dict] = None, task: Optional['ReflectionTask'] = None):
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

    @staticmethod
    def from_dictionary(d: dict) -> Type['ReflectionSolver']:
        '''Create a new reflection solver instance and initialize its parameters from a dictionary.
        The dictionary fields used are `method` (which should be among the `solver_types`) 
        and the parameter fields that are required by the selected solver method.
        '''

        if 'method' not in d or d['method'] not in ReflectionSolver.solver_types:
            raise ValueError('In ReflectionSolver.from_dictionary(): unknown solver method or no method specified')
        else:
            cls = ReflectionSolver.solver_types[d['method']]
            if cls == 'AdiabaticReflectionSolver':
                from .reflsolver_adiabatic import AdiabaticReflectionSolver
            else:
                raise ValueError(f'In ReflectionSolver.from_dictionary(): unknown reflectance solver class {cls}')
            if isinstance(cls, str) and cls in locals():
                solver = locals()[cls]()   # A kind of Java-reflection-like instantiation by name
                solver.initialize_params(d)
                return solver
            else:
                #print(f'globals = {globals()}')
                raise ValueError(f'In ReflectionSolver.from_dictionary(): unknown reflectance solver class "{cls}"')

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
