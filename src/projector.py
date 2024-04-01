# =================================================================================================
# Projector.py: a base class for projectors onto the momentum grid
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Optional, Sequence, Union, Type, Tuple, List
from numpy.typing import NDArray
import numpy as np

from .lattice import CrystalLattice
from .utils import nparr2str

class Kgrid:
    '''A class encapsulating a finite grid in the Brillouin zone of a crystal.
    '''

    lattice: Type[CrystalLattice]       #: Crystal lattice for which the momentum grid is defined
    npoints: NDArray[np.int]            #: Number of k grid points along each reciprocal lattice vector
    shift:   NDArray[np.float]          #: Additional shift of k points from Gamma
    type:    Union[str, Sequence[str]]  #: Type of grid, such as uniform, containing Gamma, etc.

    _supported_types = { 'MP' }

    def __init__(self, *,
                 lattice: Type[CrystalLattice], 
                 type: Optional[str] = 'MP', 
                 npoints: NDArray[np.int], 
                 shift: Optional[NDArray[np.float]] = None
                ):
        self.lattice = lattice
        if type not in self._supported_types:
            raise ValueError(f'In Kgrid.__init__(): unsupported grid type {type}')
        self.type = type
        self.npoints = np.array(npoints, dtype=int)
        self.shift = np.zeros(self.npoints.shape[0], dtype=float) if shift is None else np.array(shift, dtype=float)
        # Make some consistency checks
        if self.shift.shape[0] != self.npoints.shape[0] or self.npoints.shape[0] != self.lattice.crystal_dimension():
            raise ValueError('In Kgrid.__init__(): dimension mismatch')
    
    def size(self) -> int:
        'Returns the total number of points in the grid'
        if self.lattice is None:
            return 0
        else:
            return np.prod(self.npoints)
    
    def point_index_scalar2vector(self, i: int) -> NDArray[np.int]:
        '''For the i-th grid point, returns a set of its zero-based indices (i_1, i_2, ..., i_d) 
        along each grid dimension. By definition, i = i_1 + i_2 * n_1 + i_3 * (n_1 * n_2) + ...
        '''
        ndim = self.npoints.shape[0]
        idx = np.zeros(ndim, dtype=int)
        ii = i % self.size()
        for proj in range(ndim):
            idx[proj] = ii % self.npoints[proj]
            ii //= self.npoints[proj]
        return idx
    
    def point_index_vector2scalar(self, idx: NDArray[np.int]) -> int:
        'For a grid point with zero-based indices (i_1, i_2, ..., i_d), returns its scalar index i in the grid'
        ndim = self.npoints.shape[0]
        i = 0
        for proj in range(ndim - 1, -1, -1):
            i += idx[proj]
            if proj != 0:
                i *= self.npoints[proj - 1]
        return i

    def grid_point_fractional(self, i: int) -> NDArray[np.float]:
        'Returns fractional coordinates of i-th k point in the grid'
        return self.shift + self.point_index_scalar2vector(i) / self.npoints
    
    def grid_point_Cartesian(self, i: int) -> NDArray[np.float]:
        'Returns Cartesian coordinates of i-th k point in the grid'
        return self.lattice.fractional_coords_to_reciprocal_vector(self.grid_point_fractional(i))

    def __getitem__(self, i: int) -> NDArray[np.float]:
        'Returns i-th k point in the grid, in Cartesian coordinates'
        return self.grid_point_Cartesian(i)
    
    class _Iterator:
        'Iterator of k points in the given grid'
        grid:       'Kgrid'
        i:          int
        imax:       int
        Cartesian:  bool

        def __init__(self, grid: 'Kgrid', Cartesian: bool = False):
            self.grid = grid
            self.i = 0
            self.imax = grid.size()
            self.Cartesian = Cartesian

        def __iter__(self):
            return self

        def __next__(self) -> NDArray[np.float]:
            if self.i >= self.imax:
                raise StopIteration()
            k = self.grid.grid_point_Cartesian(self.i) if self.Cartesian else self.grid.grid_point_fractional(self.i)
            self.i += 1
            return k

    def kpoints(self, Cartesian: Optional[bool] = True) -> 'Kgrid._Iterator':
        '''Provides an iterator over all grid points. 
        `Cartesian` = True or False requests access to Cartesian or fractional coordinates of the k points.
        '''
        return Kgrid._Iterator(self, Cartesian=Cartesian)
    
    def __iter__(self) -> 'Kgrid._Iterator':
        'Provides an iterator over Cartesian coordinates of all grid points.'
        return self.kpoints(Cartesian=True)
    
    def __str__(self) -> str:
        s = f'Kgrid(type={self.type}, npoints={list(self.npoints)}'
        if np.linalg.norm(self.shift) > 1e-3:
            s += ', shift=' + nparr2str(self.shift, prec=3, fmt='fixed')
        return s + ')'

    def closest_point_index(self, k: NDArray[np.float], 
                            Cartesian: Optional[bool] = False, 
                            scalar_index: Optional[bool] = False) -> Union[int, NDArray[np.int]]:
        '''Returns a scalar index `i` or a vector index `(i_1, ..., i_d) of a grid point closest to `k`,
        the latter given in fractional or Cartesian coordinates. The `Cartesian` and `scalar_index` flags
        control the formats of `k` and the returned index.
        '''
        k_frac = k if not Cartesian else self.lattice.reciprocal_vector_to_fractional_coords(k)
        idx_vec = np.array(np.round(np.mod(k_frac, 1) * self.npoints), dtype=int)
        return idx_vec if not scalar_index else self.point_index_vector2scalar(idx_vec)


class Reflector:
    '''A class encapsulating reflection operations on a density matrix rho_mn(k) (diagonal in k space)
    for a fixed boundary normal vector. The k space is 
    '''

    # The reflection amplitudes are stored as a sparse matrix, i.e., as a list of in- and out-states.
    # Namely, per each k in the grid and each band n, we store the in-state wave function and 
    # the output state(s) (k_m, A_m u_m).
    reflection_amps: List[List[Tuple[NDArray[np.float], NDArray[np.complex]]]]
    kgrid:           Kgrid

    def reflect_density_matrix(self, rho_k: NDArray[np.complex]) -> NDArray[np.complex]:
        '''Apply reflection to a density matrix rho_k (a complex array with shape = (Nk, Nbands, NBands)).
        Returns an array of the same shape, in which the incident-state contirbutions of rho_k are retained, 
        and the reflected-state contibutions are constructed from the known reflection amplitudes.
        '''
        for ik, k in self.kgrid.kpoints(Cartesian=False):
            for kprime, uprime in self.reflection_amps[ik]:
                ikprime = self.kgrid.closest_point_index(kprime, Cartesian=False, scalar_index=True)
                # TODO: transport uprime to its counterpart at the grid point
                # ...


class KgridProjector:
    pass
