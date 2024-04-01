# =================================================================================================
# ElecStructure.py: import/export of periodic-DFT electronic structure & its manipulation
# -------------------------------------------------------------------------------------------------
# A special focus here is on retrieving states with a given energy implicitly from the dispersion
# relation E_n(\vec{k}) and on a "parallel transport" of bands along a specified direction in
# the Brillouin zone.
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Optional, Sequence, List, Tuple, Set, Union, Type
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import math
import numpy as np

from .lattice import CrystalLattice
from . import units
from .utils import nparr2str

# -------------------------------------------------------------------------------------------------
# Bloch states further allowed to be adiabatically transported across the Brillouin zone
# -------------------------------------------------------------------------------------------------

class BlochState:
    '''An electronic state of the form u_kn(\\vec{x}) exp(i\\vec{k}\\vec{x})
    with crystal momentum \\vec{k} in BZ and belonging to a band number n.
    The spin quantum number, if any, is assumed to be included into n.
    '''
    k_frac:    NDArray[np.float]    # Crystal momentum in fractional coordinates
    n:         int                  # Band number
    E:         float                # Energy, in ergs
    u:         NDArray[np.complex]  # Bloch wave function, a normalized complex column vector of H_band(k)
    v_band:    NDArray[np.float]    # A Cartesian band velocity vector
    
    lattice:   CrystalLattice  # A reference to the lattice for which the electronic structure was found

    def __init__(self, *, 
                 k_frac: Sequence[float] = [0.0, 0.0, 0.0], 
                 n: int = -1,
                 E: float = math.nan,
                 u: Sequence[complex] = None,
                 v_band: Sequence[float] = None,
                 lattice: CrystalLattice = None,
                 k_cart: Sequence[float] = None):
        '''Construct a Bloch state with a crystal momentum \\vec{k} and Bloch function u:
            * `k_frac` and `k_cart` define the Cartesian or fractional coordinates of \\vec{k},
            * `n` is the band number,
            * `E` is the energy of the state,
            * `u` is the normalized wave function (e.g., expansion coeffs in the Wannier functions w_ki),
            * `v_band` is the group (or band) velocity, a Cartesian 3D vector
            * `lattice` defines a reference crystal lattice.
        All fields are, in principle, optional, but certain operations are inaccessible without 
        the corresponding fields initialized.
        '''
        self.lattice = lattice
        if k_cart != None:
            if not lattice:
                raise ValueError('To construct a BlochState from a Cartesian k vector, a lattice should be specified')
            self.k_frac = lattice.reciprocal_vector_to_fractional_coords(k_cart)
        else:
            self.k_frac = np.array(k_frac, dtype=float)
        self.n, self.E = n, E
        self.u = None if u is None else np.array(u) # dtype=complex)
        self.v_band = None if v_band is None else np.array(v_band, dtype=float)

    @property
    def k_cart(self) -> NDArray[np.float]:
        "The crystal momentum in Cartesian coordinates"
        if not self.lattice:
            raise ValueError('To access a Cartesian k vector of a BlochState, the latter should be associated with a lattice')
        return self.lattice.fractional_coords_to_reciprocal_vector(self.k_frac)
    
    @k_cart.setter
    def k_cart(self, k_cart_value: Sequence[float]) -> NDArray[np.float]:
        "The crystal momentum in Cartesian coordinates"
        if not self.lattice:
            raise ValueError('To access a Cartesian k vector of a BlochState, the latter should be associated with a lattice')
        self.k_frac = self.lattice.reciprocal_vector_to_fractional_coords(k_cart_value)

    def vband_normal(self) -> float:
        "The band velocity along the outer normal to a crystalline half-space (if initialized with one)"
        if not self.lattice or not hasattr(self.lattice, 'boundary_normal'):
            raise ValueError('In BlochState.vband_normal(): cannot evaluate result without the boundary normal vector')
        return np.dot(self.v_band, self.lattice.boundary_normal())

    # For backward compatibility with a tuple of the state's properties
    def as_tuple(self) -> Tuple[NDArray[np.float], int, float, NDArray[np.complex], NDArray[np.float]]:
        "Returns a tuple representation in the form (\\vec{k}_fractional, n, E, u, v_band)"
        return (self.k_frac, self.n, self.E, self.u, self.v_band)
    
    def __getitem__(self, i: int) -> Union[NDArray, int, float]:
        '''Returns a state property number i, corresponding to the indices in a tuple
        (\\vec{k}_fractional, n, E, u, v_band). If the field is not set, returns None.
        '''
        return self.as_tuple()[i]    # A dummy implementation for compatibility
    
    def __setitem__(self, i: int, field_value: Union[Sequence[float], Sequence[complex], int, float]) -> None:
        '''Assigns a new value to a state property number i, corresponding to the indices in a tuple
        (\\vec{k}_fractional, n, E, u, v_band).
        '''
        if i == 0:
            self.k_frac = np.array(field_value, dtype=float)
        elif i == 1:
            self.n = int(field_value)
        elif i == 2:
            self.E = float(field_value)
        elif i == 3:
            self.u = np.array(field_value) # dtype=complex
        elif i == 4:
            self.v_band = np.array(field_value, dtype=float)
        else:
            raise ValueError('Field index out of range')

    def __str__(self) -> str:
        return f'BlochState(k_frac={nparr2str(self.k, prec=3)}, n={self.n}, E={self.E / units.eV: <.3f}eV)'
# end(class BlochState)

# -------------------------------------------------------------------------------------------------
# Encapsulation of the bulk electronic structure
# -------------------------------------------------------------------------------------------------

class ElectronicStructure(ABC): # abstract class
    '''A base class encapsulating bulk electronic structure of a crystalline material, its import/export,
    and manipulations with bands and electronic states. Specific types of the structure representation,
    such as tight-binding model, Bloch bands, or their Wannierized version, are to be defined in derived
    classes.
    '''

    lattice: CrystalLattice                     #: The underlying lattice parameters

    # Derived electronic structure classes supported by from_dictionary() method
    estructure_types = { 'Wannierized': 'WannierElectronicStructure' }

    def __init__(self, *, lattice: CrystalLattice = None): 
        self.lattice = lattice

    def load(self, **kwargs): 
        "Load or initialize the electronic structure with list of parameters (or data files)"
        pass

    @staticmethod
    def from_dictionary(d: dict) -> Type['ElectronicStructure']:
        '''Create a new electronic structure instance and initialize its parameters from a dictionary.
        The dictionary fields used are `type` (which should be among the `estructure_types`) 
        and the parameter fields that are required by the selected solver method.
        '''
        if 'type' not in d or d['type'] not in ElectronicStructure.estructure_types:
            raise ValueError('In ElectronicStructure.from_dictionary(): unknown electronic structure data type or no type specified')
        cls = ElectronicStructure.estructure_types[d['type']] # Should be a type or a str
        if isinstance(cls, type):
            struc = cls()
        elif isinstance(cls, str) and cls in globals():
            struc = globals()[cls]()   # A kind of Java-reflection-like instantiation by name
        else:
            raise ValueError('In ElectronicStructure.from_dictionary(): unknown electronic structure class')
        struc.initialize_from_dictionary(d)
        return struc
    
    def initialize_from_dictionary(self, d: dict) -> None:
        '''Initialize electronic structure from a set of key-value pairs d, e.g., one retrieved from 
        a configuration file
        '''
        pass

    def __str__(self) -> str:
        s = f'{type(self).__name__}('
        if not self.lattice is None:
            s += f'lattice={self.lattice}'
        return s + ')'

    @abstractmethod
    def electronic_states_BZ(self, k_fractional: NDArray[np.float], *, 
                             return_value: str = 'eigenstates'
                            ) -> Union[List[BlochState], Tuple[List[BlochState], NDArray]]:
        '''Find the electronic states for a k vector defined by its fractional coordinates.
        The bands are listed in the ascending-energy order. If `return_value` == "eigenstates",
        only the states are returned; if `return_value` == 'eigenstates_Hband', then a 2-tuple
        is returned, with the 2nd element containing the band Hamiltonian matrix.
        '''
        pass


    def propagate_band_in_BZ(self, state_k: BlochState, kprime_frac: NDArray[np.float]) -> list:
        '''For a state_k = (\\vec{k}, n, E_n, u_n, v_n) with crystal momentum \\vec{k}, 
        find a state corresponding to the same band at a nearby momentum \\vec{kprime_frac}.
        Both momenta are given in fractional coordinates, and are assumed to be sufficiently close,
        otherwise a band crossing may be left unnoticed.
        '''
        # Strategies: (1) energies are close; 
        #             (2) band numbers are close; 
        #             (3) Bloch function are close to proportional
        _implementation_version_ = 0
        if _implementation_version_ == 0: 
            # Version 0: Choose the state with the "most collinear" wave function 
            states_kprime = self.electronic_states_BZ(kprime_frac, return_value='eigenstates')
            u_kprime = np.array([state.u for state in states_kprime]).T
            return states_kprime[np.argmax(np.abs(state_k.u.conj() @ u_kprime))]
        elif _implementation_version_ == 0: 
            # Version 1: The same strategy but making infinitesimal steps from k to k'
            H = self.electronic_states_BZ(state_k.k_frac, return_value='eigenstates_Hband')[1]
            states_kprime, Hprime = self.electronic_states_BZ(kprime_frac, return_value='eigenstates_Hband')
            E_prime = np.array([state.E for state in states_kprime])
            u_prime = np.array([state.u for state in states_kprime]).T
            # Find deltaH to measure the scale of deltaE
            deltaH = np.linalg.norm(Hprime - H, 'fro') # / sqrt(H.shape[0])
            
            dE_smallest = np.min(np.abs(state_k.E - E_prime)) # The closest band at k'
            nsegments = 5 if deltaH > 5 * dE_smallest else math.ceil(deltaH / dE_smallest)
            u_n = np.array(state_k.u)
            for i in range(1, nsegments + 1):
                H_nplus1 =  (1 - i / nsegments) * H + (i / nsegments) * Hprime
                E_nplus1, u_nplus1 = np.linalg.eigh(H_nplus1)
                # Choose the state with the closest eigenfunction, as in Version 0
                u_n = u_nplus1[:, np.argmax(np.abs(u_n.conj() @ u_nplus1))]
            return states_kprime[np.argmax(np.abs(u_n.conj() @ u_prime))]
        elif _implementation_version_ == 10:
            # Version 10: Choose the state that is the closest in energy
            H = self.electronic_states_BZ(state_k.k_frac, return_value='eigenstates_Hband')[1]
            states_kprime, Hprime = self.electronic_states_BZ(kprime_frac, return_value='eigenstates_Hband')
            E_prime = np.array([state.E for state in states_kprime])
            u_prime = np.array([state.u for state in states_kprime]).T
            deltaE_estimation = state_k.u.conj().dot((Hprime - H).dot(state_k.u)).real
            # Indices of the states that are sufficiently close in energy and sufficiently "collinear" to the reference one
            indices = [i for i in range(len(states_kprime)) 
                       if abs(states_kprime[i] - state_k.E) < 3.0 * abs(deltaE_estimation)
                       and abs(states_kprime[i].conj().dot(state_k.u)) > 0.5
                      ] 
            if len(indices) == 0:    # All states are too far: return the one that is closest in energy
                return states_kprime[np.argmin([abs(state.E - state_k.E) for state in states_kprime])]
            else:                    # Amongst the screened states, choose the one that is closest in energy
                return states_kprime[np.argmin([abs(states_kprime[i].E - state_k.E) for i in indices])]
        else:
            raise NotImplementedError(f'In ElectronicStructure.propagate_band_in_BZ(): unknown _implementation_version_ = {_implementation_version_}')
        

    def propagate_band_in_energy(self, state_k: BlochState, Eprime: float, deltak_direction: NDArray[np.float], 
                                 options: dict = {'E_tolerance': 0.0001 * units.eV, 'max_iterations': 50}
                                ) -> BlochState:
        '''For a state_k = (\\vec{k}, n, E_n, u_n, v_n) with crystal momentum \\vec{k}, 
        find the state corresponding to the same band and energy Eprime at a nearby momentum of the form
        \\vec{kprime} = \\vec{k} + q \\vec{deltak_direction} (all momenta are in fractional coordinates)
        `Eprime` is assumed sufficiently close to E_n, otherwise a band crossing may be left unnoticed. 
        `options` is a set of implementation-specific options that may affect the calculation; 
        the most important entry is 'E_tolerance', the output energy tolerance. 
        This argument should be used with care, as it may limit the class polymorphism.
        Returns the nearby state_kprime as a BlochState object. 
        '''
        _implementation_version_ = 0
        if _implementation_version_ == 0:
            # Version 0 (): a general, purely numerical root finder 
            # [virtually, a Newton's method + propagation along the band]
            # Can be overridden in derived classes
            E_tolerance = options.get('E_tolerance', 0.0001 * units.eV)
            max_iterations = options.get('max_iterations', 50)
            deltak_dir_Cart = self.lattice.fractional_coords_to_reciprocal_vector(deltak_direction)
            
            iter = 0
            current_state = state_k
            q = 0
            while iter <= max_iterations and abs(current_state.E - Eprime) > E_tolerance:
                # Estimate the value of q' from a linear equation: E(q') ~ E(q) + E'(q) * (q'-q) = Eprime,
                # where E'(q) = hbar \vec{v}(q) d\vec{k}_cart/dq = hbar \vec{v}(q) frac2cart(\vec{deltak_direction})
                q_next = (Eprime - current_state.E) / (units.hbar * np.dot(current_state.v_band, deltak_dir_Cart))
                current_state = self.propagate_band_in_BZ(current_state, state_k.k_frac + q_next * np.array(deltak_direction))
                iter += 1
            return current_state if abs(current_state.E - Eprime) < abs(state_k.E - Eprime) else state_k
        else:
            raise NotImplementedError(f'In ElectronicStructure.propagate_band_in_energy(): unknown _implementation_version_ = {_implementation_version_}')

# -------------------------------------------------------------------------------------------------
# Bulk electronic structure described in terms of Wannier functions (a.k.a. Wannierized one)
# -------------------------------------------------------------------------------------------------
        
class WannierElectronicStructure(ElectronicStructure):
    '''A base class encapsulating bulk electronic structure of a crystalline material, its import/export,
    and manipulations with bands and electronic states. Specific types of the structure representation,
    such as tight-binding model, Bloch bands, or their Wannierized version, are to be defined in derived
    classes.
    '''

    R_points: NDArray[np.int]                        #: A list of \vec{R} vectors for each Wannier function, in "fractional" real-space coordinates (each \vec{R} is stored row-wise)
    H_wannier: NDArray[Union[np.float, np.complex]]  #: A matrix <w_0|H|w_R> for all Wannier functions 

    def load(self, **kwargs):
        '''Load or initialize the electronic structure with list of parameters (or data files).
        Among the keyword arguments `kwargs`, a format should be specified, as well as its options.
        Format(s) currently supported:
            * file.jdftx.wannierized: see documentation for load_jdftx_wannierized()
        '''
        if 'format' not in kwargs:
            raise ValueError('In WannierElectronicStructure.load(): input format should be specified')
        fmt = kwargs['format']
        if fmt == 'file.jdftx.wannierized':
            return self.load_jdftx_wannierized(**kwargs)
        else:
            raise ValueError(f'In WannierElectronicStructure.load(): unknown format "{fmt}"')
    
    def __str__(self) -> str:
        s = super().__str__()
        if s[-1] != ')' or self.H_wannier is None: 
            return s
        else:
            return s[:-1] + f', nbands={self.H_wannier.shape[1]})'
    
    def initialize_from_dictionary(self, d: dict) -> None:
        '''Initialize electronic structure from a set of key-value pairs d, e.g., one retrieved from 
        a configuration file.
        '''
        type = d.get('type', 'Wannierized')
        format = d.get('format', 'jdftx')
        dft_outfile  = d.get('filename.dft-output', None)
        wannier_stem = d.get('filename.wannier-stem', None)
        if type != 'Wannierized':
            raise ValueError(f'In WannierElectronicStructure.initialize_from_dictionary: unknown data type {type}')
        if format == 'jdftx':
            self.load_jdftx_wannierized(format='file.jdftx.wannierized',
                                        dft_outfile=dft_outfile, wannier_stem=wannier_stem)
        else:
            raise ValueError(f'In WannierElectronicStructure.initialize_from_dictionary: unsupported data format {format}')
        
    def load_jdftx_wannierized(self, *,
                               lattice_vectors: Optional[NDArray] = None,
                               dft_stem: Optional[str] = None,
                               dft_outfile: Optional[str] = None,
                               wannier_stem: Optional[str] = None,
                               wannier_cellmap_file: Optional[str] = None,
                               wannier_cellweights_file: Optional[str] = None,
                               wannier_Hamiltonian_file: Optional[str] = None,
                               format: Optional[str] = 'file.jdftx.wannierized'):
        '''Load Wannierized electronic structure previously calculated using JDFTx software.
        Either a `dft_stem`, or an `explicit dft_outfile` name should be specified to locate a DFT
        band structure calculation used for Wannierization. Analogously, if `wannier_stem` is not specified,
        then Wannierization-related filenames should be specified explicitly; otherwise, they are
        derived by adding different extensions to the stem. The only format supported is 'file.jdftx.wannierized'.
        
        The function does not return a value, but initizalizes the electronic structure parameters instead.
        '''
        if format != 'file.jdftx.wannierized':
            return ValueError(f'In WannierElectronicStructure.load_jdftx_wannierized(): unknown format "{format}"')
        # 0. Retrieve/constuct necessary filenames
        try:
            _dft_outfile = dft_outfile if dft_outfile else dft_stem + '.out'
            _wannier_cellmap_file = wannier_cellmap_file if wannier_cellmap_file else wannier_stem + '.mlwfCellMap'
            _wannier_cellweights_file = wannier_cellweights_file if wannier_cellweights_file else wannier_stem + '.mlwfCellWeights'
            _wannier_Hamiltonian_file = wannier_Hamiltonian_file if wannier_Hamiltonian_file else wannier_stem + '.mlwfH'
        except TypeError as e:
            raise ValueError('In WannierElectronicStructure.load_jdftx_wannierized(): either a common filename stem ' + 
                                'should be specified as dft_stem, or explicit names for each input file') 
        # 1. Get the lattice vectors
        if lattice_vectors != None:
            _lattice_vectors = np.array(lattice_vectors, dtype=float)
        else:
            with open(_dft_outfile, 'rt') as f:
                while True: 
                    ln = f.readline()
                    if not ln:
                        break
                    ln = ln.strip()
                    if not ln.startswith('---') or not ln.endswith('---') or 'INITIALIZING THE GRID' not in ln.upper():
                        continue
                    # Parse the lattice vectors block
                    ln = f.readline()
                    if ln.split() != ['R', '=']:
                        raise ValueError('In WannierElectronicStructure.load_jdftx_wannierized(): unknown format of lattice vectors in DFT output file')
                    _lattice_vectors = []
                    while True:
                        ln = f.readline().strip()
                        if not ln.startswith('[') or not ln.endswith(']'):
                            break
                        _lattice_vectors.append([float(x) * units.BohrRadius for x in ln.split()[1:-1]])
                    _lattice_vectors = np.array(_lattice_vectors, dtype=float)
        if self.lattice == None:
            self.lattice = CrystalLattice(lattice_vectors=_lattice_vectors)
        self.lattice.lattice_vectors = np.array(_lattice_vectors)
        # 2. Retrieve R-points for which Wannier functions are present, as well as the k-points
        self.R_points = np.loadtxt(_wannier_cellmap_file)[:, 0:3].astype(int)  # Wannier \vec{R} vectors in frac. coords
        wannier_weights = np.fromfile(_wannier_cellweights_file)
        ncells = self.R_points.shape[0]
        nbands = int(np.sqrt(wannier_weights.shape[0] / ncells))
        wannier_weights = wannier_weights.reshape((ncells, nbands, nbands)).swapaxes(1, 2)
        # Find the k-point grid dimensions and retrieve the Wannier Hamiltonian
        with open(_dft_outfile) as f:
            for line in f.readlines():
                if line.startswith('kpoint-folding'):
                    k_folding = np.array([int(tok) for tok in line.split()[1:4]])  # Parse three space-separated integers
        nkpoints = np.prod(k_folding)
        # N.B.: next line should be refactored, since it assumes dim = 3
        kstride = np.array([k_folding[1] * k_folding[2], k_folding[2], 1])
        H_reduced = np.fromfile(_wannier_Hamiltonian_file, dtype=float)
        # Note that the array can be stored as a real or a complex one, so let's take care when reshaping it
        if H_reduced.size == nkpoints * nbands * nbands:         # A real array
            H_reduced = H_reduced.reshape((nkpoints, nbands, nbands)).swapaxes(1, 2)
        elif H_reduced.size == 2 * nkpoints * nbands * nbands:   # A complex array
            H_reduced = (H_reduced[0::2] + 1j * H_reduced[1::2]).reshape((nkpoints, nbands, nbands)).swapaxes(1, 2)
        else:
            raise ValueError('In WannierElectronicStructure.load_jdftx_wannierized(): cannot determine the shape of the wannierized Hamiltonian')
        idx_reduced = np.dot(np.mod(self.R_points, k_folding[None,:]), kstride)
        self.H_wannier = units.Hartree * wannier_weights * H_reduced[idx_reduced]
        
    
    def electronic_states_BZ(self, k_fractional: NDArray[np.float], *, 
                             return_value: str = 'eigenstates'
                            ) -> Union[List[BlochState], Tuple[List[BlochState], NDArray]]:
        '''Find the electronic states for a k vector defined by its fractional coordinates.
        The bands are listed in the ascending-energy order. If `return_value` == "eigenstates",
        only the states are returned; if `return_value` == 'eigenstates_Hband', then a 2-tuple
        is returned, with the 2nd element containing the band Hamiltonian matrix.
        '''
        # 1. Find the band Hamiltonian and its derivatives
        R_points_cartesian = np.array([self.lattice.fractional_coords_to_real_vector(R) for R in self.R_points])
        exp_ikR = np.exp(2.0j * np.pi * np.dot(self.R_points, k_fractional))   # A vector of exp{i k R} for all R
        H_k = np.tensordot(exp_ikR, self.H_wannier, axes=1)
        gradH_k = 1.0j * np.tensordot(R_points_cartesian.T * exp_ikR[None,:], self.H_wannier, axes=1)
        # 2. Find the bands and their band velocities
        E, u = np.linalg.eigh(H_k)    # N.B.: the eigenvalues are in ascending-energy order
        dim = gradH_k.shape[0]
        nbands = E.shape[0]
        v_band = [(1.0 / units.hbar) * u[:,n].conj().dot(gradH_k.dot(u[:,n]).T).real for n in range(nbands)]
        eigenstates = [BlochState(k_frac=k_fractional, n=n, E=E[n], u=u[:,n], v_band=v_band[n], lattice=self.lattice) 
                       for n in range(nbands)
                      ]
        if return_value == 'eigenstates':
            return eigenstates
        elif return_value == 'eigenstates_Hband':
            return (eigenstates, H_k)
        else:
            raise ValueError('In WannierElectronicStructure.electronic_states_BZ(): unsupported return_value parameter')

