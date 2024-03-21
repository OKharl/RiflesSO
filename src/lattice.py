# =================================================================================================
# Lattice.py: crystals with boundaries and associated functions for RiflesSO
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Sequence, List, Union, Optional, Tuple
import numpy as np
import re
from .utils import lcm_list, reduce_to_coprimes, normalize, solve_LDE_3vars_homo, snap_to_rational
from . import utils   # For re_* RegEx constants
from . import units


class CrystalLattice:
    '''Encapsulates a periodic Bravais lattice filling a d-dimensional half-space (d=1,2,3) 
    embedded into the n-dimensional target space (n >= d), as well as the corresponding reciprocal space'''

    _lattice_vectors: np.ndarray       #: d lattice vectors as columns of a matrix (shape = (n,d))
    _reciprocal_vectors: np.ndarray    #: d reciprocal vectors as rows of matrix (shape = (d,n))

    def __init__(self, *, 
                 lattice_vectors:  Sequence[float] = np.identity(3, dtype=float)
                ):
        '''Constructs a CrystalLattice object with d given primitive vectors, 
        stored column_wise in lattice_vectors. Note: lattice_vectors.shape = (n, d), n >= d.
        '''
        self.lattice_vectors = lattice_vectors # Also initializes reciprocals

    def metric_tensor(self) -> np.ndarray:
        "Returns a (dxd) matrix g_ab = <e_a|e_b>, where e_a is the lattice vector No.a"
        return self._lattice_vectors.T.dot(self._lattice_vectors)

    @property
    def reciprocal_vectors(self) -> np.ndarray:
        "Primitive vectors of the reciprocal vectors stored row-wise, in inverse length units"
        return self._reciprocal_vectors
    
    @reciprocal_vectors.setter
    def reciprocal_vectors(self, new_vecs):
        self._reciprocal_vectors = np.array(new_vecs, dtype=float)
        # Note: if d < n (say, 2D graphene in 3D space), direct inversion of reciprocal_vectors is impossible
        g = self._reciprocal_vectors.dot(self._reciprocal_vectors.T)
        self._lattice_vectors = (2 * np.pi) * (self._reciprocal_vectors.T @ np.linalg.inv(g))

    @property
    def lattice_vectors(self) -> np.ndarray:
        "Primitive lattice vectors stored column-wise, in length units"
        return self._lattice_vectors
    
    @lattice_vectors.setter
    def lattice_vectors(self, new_vecs):
        self._lattice_vectors = np.array(new_vecs, dtype=float)
        # Note: if d < n (say, 2D graphene in 3D space), direct inversion of lattice_vectors is impossible
        self._reciprocal_vectors = (2 * np.pi) * np.linalg.inv(self.metric_tensor()) @ self._lattice_vectors.T

    def crystal_dimension(self) -> np.ndarray:
        "Returns d, the dimensionality of the crystal (e.g., 1 for polyacetylene, 2 for graphene, 3 for graphite)"
        return self._lattice_vectors.shape[1]

    def target_space_dimension(self) -> np.ndarray:
        "Returns n, the dimensionality of space the crystal is embedded into. N.B.: n >= crystal_dimension()"
        return self._lattice_vectors.shape[0]

    def fractional_coords_to_real_vector(self, coords: Sequence) -> np.ndarray:
        "Transforms a list of d coordinates x_i to an nD vector x_i \\vec{e}_i in real space, where \\vec{e}_i are the primitive vectors"
        return self._lattice_vectors.dot(coords)

    def fractional_coords_to_reciprocal_vector(self, coords: Sequence) -> np.ndarray:
        "Transforms a list of d coordinates k_i to an nD vector k_i \\vec{f}_i in the reciprocal space with primitive vectors \\vec{f}_i"
        return np.dot(coords, self._reciprocal_vectors)

    def real_vector_to_fractional_coords(self, x: Sequence) -> np.ndarray:
        '''Transforms an nD vector \\vec{x} in real space into its d coordinates in the primitive vector basis. 
        Note that if d < n and \\vec{x} does not lie in the crystal plane, the out-of-plane coordinates are lost'''
        return (0.5 / np.pi) * self._reciprocal_vectors.dot(x)

    def reciprocal_vector_to_fractional_coords(self, k: Sequence) -> np.ndarray:
        '''Transforms an nD momentum vector \\vec{k} in space into its d coordinates in the reciprocal vector basis. 
        Note that if d < n and \\vec{k} does not lie in the crystal plane, the out-of-plane coordinates are lost.
        '''
        return (0.5 / np.pi) * np.dot(k, self._lattice_vectors)
    
    def project_onto_crystal(self, vec: Sequence[float]) -> np.ndarray:
        "Project an arbitrary vector onto the crystal plane (if it is, e.g. two-dimensional in a 3D space)"
        if self.target_space_dimension() == self.crystal_dimension():
            return np.array(vec)     # Essentially nothing to project
        else: 
            return self.fractional_coords_to_real_vector(self.real_vector_to_fractional_coords(vec))
    
    def load(self, filename: str, format: Union[str, Tuple[str, dict]]) -> None:
        '''Load the crystal structure from a file `filename`, such as an output file 
        written by a DFT software. `format` can be either a string, such as \"jdftx.dft_outfile\",
        or a tuple of the form ("file_name", {"param1": param1_value, "param2": param2_value}),
        with the parameters defining further options, e.g., for parsing a specific section of the file.
        '''
        if format == 'jdftx.dft_outfile':
            self._load_jdftx_dft_outfile(filename)
        else:
            raise ValueError(f'CrystalLattice.load() does not support format {format}')

    def _load_jdftx_dft_outfile(self, filename: str, occurrence: int = 0):
        '''A branch of load() function for JDFTx output files of DFT calculations. 
        If occurrence = 0, reads the first occurrence of the lattice vectors in the file and stops;
        if occurrence = 1, 2, 3, ..., searches uses the corresponding nth set of lattice vectors;
        if occurrence = -1, uses the last set of the lattice vectors.
        '''
        sep, ws, ow = utils.re_separator('-', 3), utils.re_whitespace, utils.re_opt_whitespace
        occurred = 0
        with open(filename, 'rt') as f:
            while occurrence == -1 or occurred <= occurrence:
                ln = utils.find_line_in_file(f, 
                                             [[sep + ow + 'Initializing the grid' + ow + sep,
                                               'R' + ow + r'\='
                                              ]
                                            ], 
                                            re.IGNORECASE)
                if ln == None:
                    if occurrence == -1 and occurred > 0:
                        break
                    raise ValueError('In CrystalLattice._load_jdftx_dft_outfile(): cannot find requested lattice vectors in DFT output file')
                occurred += 1
                lattice_vectors = []
                while True:
                    ln = f.readline().strip()
                    m = re.match('\\[' + ow + utils.re_real + '(?:' + ws + ')*' + '\\]', ln)
                    if not m:
                        if lattice_vectors == []:
                            raise ValueError('In CrystalLattice._load_jdftx_dft_outfile(): unknown format of lattice vectors in DFT output file')
                        break
                    else:
                        lattice_vectors.append([float(x) * units.BohrRadius for x in ln.split()[1:4]])
        self.lattice_vectors = lattice_vectors



class CrystallineHalfSpace(CrystalLattice):
    '''Encapsulates a periodic Bravais lattice filling a d-dimensional half-space (d=1,2,3) 
    embedded into the n-dimensional target space (n >= d), as well as the corresponding reciprocal space'''

    _lattice_vectors: np.ndarray       #: d lattice vectors as columns of a matrix (shape = (n,d))
    _reciprocal_vectors: np.ndarray    #: d reciprocal vectors as rows of matrix (shape = (d,n))
    boundary_plane: Union[List[int], np.ndarray]   #: either (integer) Miller indices or frac. coords of \vec{n} in the reciprocal basis


    def __init__(self, *, 
                 lattice_vectors:        Sequence[float]      = np.identity(3, dtype=float), 
                 boundary_plane_normal:  Optional[np.ndarray] = None,
                 boundary_plane_indices: Optional[List[int]]  = None
                ):
        '''Constructs a CrystallineHalfSpace object with the following properties:
            lattice_vectors:        a set of column-wise lattice vectors (shape=(n, d), n >= d),
            boundary_plane_normal:  an outward normal vector to the boundary [optional],
            boundary_plane_indices: Miller indices [h,k,l] of the boundary plane [optional]
        '''
        super().__init__(lattice_vectors)
        if boundary_plane_normal != None:
            self.boundary_plane = self.reciprocal_vector_to_fractional_coords(boundary_plane_normal)
        elif boundary_plane_indices != None:
            self.boundary_plane = reduce_to_coprimes(boundary_plane_indices)
        else: # [0, 0, 1] for 3D crystals, [0, 1] for 2D ones
            dim = self.crystal_dimension()
            if dim == 1:
                self.boundary_plane = []
            else:
                self.boundary_plane = [0] * dim
                self.boundary_plane[-1] = 1

    def is_boundary_normal_rational(self) -> bool:
        "Returns True if the crystal boundary is rational, i.e., corresponds to certain Miller indices (hkl)"
        return isinstance(self.boundary_plane, list) 
               # and list(map(type, self.boundary_plane)) == [int] * len(self.boundary_plane)

    def boundary_normal(self) -> np.ndarray:
        "Returns a unit n-dimensional vector describing the outer normal to the crystal boundary"
        return normalize(self.fractional_coords_to_reciprocal_vector(self.boundary_plane))
        
    def snap_to_rational_boundary(self, delta_normal_max: float = 0.05) -> List[int]:
        '''Replace the normal vector currently installed to the 'rational' one (hkl),
        such that max(h,k,l) --> min and |new_normal - current_normal| <= delta_normal_max.
        Returns the Miller indices of the new normal vector.
        '''
        if self.is_boundary_normal_rational():
            return self.boundary_plane
        # "Dummy" Version 0.1: just round the Miller indices to withion delta_normal_max and then find a common denominator
        boundary_plane_rounded = [snap_to_rational(n_i, delta_normal_max) for n_i in self.boundary_plane]
        common_denom = lcm_list([f.denominator for f in boundary_plane_rounded])
        self.boundary_plane = [f.numerator * (common_denom // f.denominator) for f in boundary_plane_rounded]
        return self.boundary_plane


    def boundary_lattice_vectors(self) -> np.ndarray:
        '''Find the shortest real lattice vectors, G in 2D and G_1, G_2 in 3D, that connect
        the lattice points lying at the boundary. In other words, these vectors are the periods 
        of the boundary of the semi-infinite crystal. 
        Return value: the coordinates of the vectors as rows of a 2D numpy.array[int],
        e.g. e_1 + 2 e_2 - e_3 corresponds to a row [1,2,-1]. In 1D, an empty set is returned.
        '''
        dim = self.crystal_dimension()
        if dim == 1:
            return np.array([], dtype=int)
        if not self.is_boundary_normal_rational():
            raise ValueError('In CrystallineHalfSpace.boundary_lattice_vectors(): result undefined for ' +
                             'irrational plane slopes. Consider trying to snap_to_rational_boundary() first.')
        elif dim == 2:
            h, k = self.boundary_plane
            xi_coords = np.array([k, -h], dtype=int)
            # Make the period vector xi form a right-handed basis together with the normal
            third_direction = np.cross(self._lattice_vectors[:,0], self._lattice_vectors[:,1])
            xi_third_direction = np.cross(self.fractional_coords_to_real_vector(xi_coords), self.boundary_normal())
            return xi_coords[None,:] if xi_third_direction.dot(third_direction) > 0 else -xi_coords[None,:]
        elif dim == 3:
            # Finding the basis vectors \vec{v} in the plane amounts to solving a linear Diophantine equation \vec{v} \cdot \vec{n} = 0
            xi_coords, eta_coords = solve_LDE_3vars_homo(*self.boundary_plane)
            # Make the period vectors xi, eta form a right-handed basis together with the normal
            xieta_orientation = np.cross(self.fractional_coords_to_real_vector(xi_coords), 
                                         self.fractional_coords_to_real_vector(eta_coords)
                                        ).dot(self.boundary_normal())
            return np.array([xi_coords, eta_coords]) if xieta_orientation > 0 else -np.array([xi_coords, eta_coords])
    
    def boundary_potential_kdirection(self) -> np.array:
        '''Returns a tangent vector to a line in reciprocal space, which is a locus of nonzero matrix elements of the boundary potential.
        The potential is assumed to be of the form of an increasing function V(\\vec{n} \cdot \\vec{x}), 
        where \\vec{n} is the outward normal to the boundary.
        The tangent vector is returned as a set of its components in the reciprocal vectors (a.k.a. fractional coordinates).
        If the boundary plane is given in terms of the Miller indices, a rational vector is retured; otherwise a real-valued one.
        '''
        return np.array(self.boundary_plane)  # Just that easy!..


if __name__ == '__main__':
    lat = CrystalLattice()
    lat.load('../../jdftx_calculations/graphene/scf_fine.out', 'jdftx.dft_outfile')
    print(lat.lattice_vectors)