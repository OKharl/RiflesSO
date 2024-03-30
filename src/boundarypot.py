# =================================================================================================
# BoundaryPot.py: implementation of different shapes of the bounary potential
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Optional, Type
import math
from scipy.optimize import root_scalar

from . import units
from . import utils


class BoundaryPotential:
    '''Encapsulation of the boundary potential energy V(z) confining electrons to a finite crystalline sample,
    where z = \\vec{x} ⋅ \\vec{n} is normal to the boundary and z → -∞ corresponds to the bulk crystal.
    '''
    _shape: str               #: Type of the potential, for logging usage
    V_0: float                #: Height of the potential, namely, V(+∞) - V(-∞)

    def __init__(self, shape: str, V_0: float):
        "Create a BoundaryPotential with a total height V_0 and a given shape"
        self._shape = shape
        self.V_0 = V_0

    def potential_value(self, z: float) -> float:
        '''Returns the boundary potential energy V(z), z being the outward directed Cartesian coordinate.
        This is a virtual function to be overridden in derived classes; its default behavior is dummy.
        '''
        return 0.0 if z < 0.0 else self.V_0
        
    def potential_inv_function(self, V_z: float) -> float:
        '''Returns the coordinate z, for which the boundary potential V(z) = V_z. 
        Returns -∞ if V_z <= V(-∞) and +∞ if V_z > V(+∞).
        This is a virtual function to be overridden in derived classes; 
        its default behavior is a rather slow numerical inverse of the assumedly monotonic V(z)
        '''
        z_cutoff = 100.0 * units.Angstrom
        if V_z <= self.potential_value(-z_cutoff): 
            return -math.inf
        elif V_z >= self.potential_value(z_cutoff):
            return math.inf
        else:
            return root_scalar(self.potential_value, method='bisect', bracket=(-z_cutoff, z_cutoff))

    def __str__(self) -> str:
        return f'{type(self).__name__}(shape={self.shape}, V_0={self.V_0 / units.eV: <.2f}eV)'
    
    @property
    def shape(self) -> str:
        "The shape of the potential"
        return self._shape
    
    @shape.setter
    def shape(self, new_shape: str):
        "Modify the shape of the potential; may be modified in derived classes"
        self._shape = new_shape

    def initialize_from_dictionary(self, d: dict):
        'Initialize the potential shape and parameters from a set of options organized in a dictionary'
        self.shape = d['shape']
        d_height = d.get('height', 10.0)
        if isinstance(d_height, float):
            self.V_0 = d_height * units.eV
        elif isinstance(d_height, str):
            try:
                self.V_0 = utils.float_with_units(d_height, default_units='eV', allowed_units={'eV', 'Ha', 'Ry'})
            except:
                raise ValueError('In BoundaryPotential.initialize_from_dictionary(): syntax error in height field')
        else:
            raise ValueError('In BoundaryPotential.initialize_from_dictionary(): syntax error in height field')

    @classmethod
    def from_dictionary(cls, d: dict) -> Type['BoundaryPotential']:
        '''Initialize a BoundaryPotential object from a dictionary, e.g., upon parsing of JSON or YAML.
        `section` defines a root node of the dictionary, e.g., section = "params.pot" uses d["params"]["pot"]
        to extract the necessary data.
        '''
        shape = d.get('shape', 'kink.tanh')
        if shape in {'kink.tanh', 'kink.piecewise_linear'}:
            obj = KinkPotential(shape=shape, V_0 = 10 * units.eV, V_width = 10.0 * units.Angstrom)
        else:
            raise ValueError(f'In BoundaryPotential.from_dictionary(): unsupported potential shape "{shape}"')
        obj.initialize_from_dictionary(d)
        return obj


class KinkPotential(BoundaryPotential):
    '''A smooth, monotonic, and analytically invertible boundary potential;
    the classic shape is V(z) = 0.5 V_0 (1 + tanh(z / V_width)).
    '''

    # Possible values of _shape_code
    _shape_tanh             = 10
    _shape_piecewise_linear = 20
    _shape_codes            = { 'kink.tanh': _shape_tanh, 'kink.piecewise_linear': _shape_piecewise_linear } 

    _shape_code: int        # A somewhat faster version of `shape` using an integer instead of a string
    V_width: float          # A width of the potential wall

    def __init__(self, *, shape: Optional[str] = 'kink.tanh', V_0: float, V_width: float):
        '''Create a KinkPotential of a given shape with a total height V_0 [energy units] 
        and width V_width [length units]. The shapes currently supported are 'tanh' and 'piecewise_linear'.
        '''
        super().__init__(shape, V_0)
        self.V_width = V_width
        self.shape = shape
        
    @BoundaryPotential.shape.setter
    def shape(self, new_shape: str):
        "Modify the shape of the potential. Supported shapes are 'kink.tanh' and 'kink.piecewise_linear'"
        if new_shape not in self._shape_codes:
            raise ValueError(f'In KinkPotential: potential shape should be in {self._shape_codes.keys()}')
        self._shape_code = self._shape_codes[new_shape]
        self._shape = new_shape
    
    def potential_value(self, z: float) -> float:
        "Returns the boundary potential energy V(z), z being the outward directed Cartesian coordinate."
        if self._shape_code == self._shape_tanh:
            return 0.5 * self.V_0 * (1 + math.tanh(z / self.V_width))
        elif self._shape_code == self._shape_piecewise_linear:
            if z <= -0.5 * self.V_width:
                return 0.0
            elif z >= 0.5 * self.V_width:
                return self.V_0
            else:
                return self.V_0 * (0.5 + z / self.V_width)
        else:
            raise NotImplementedError(f'In KinkPotential.potential_value(): unsupported potential shape {self._shape_code}')

    def potential_inv_function(self, V_z: float) -> float:
        '''Returns the coordinate z, for which the boundary potential V(z) = V_z. 
        Returns -∞ if V_z <= V(-∞) and +∞ if V_z > V(+∞)
        '''
        if V_z <= 0:
            return -math.inf
        elif V_z >= self.V_0:
            return math.inf
        elif self._shape_code == self._shape_tanh:
            return self.V_width * math.atanh(2.0 * V_z / self.V_0 - 1)
        elif self._shape_code == self._shape_piecewise_linear:
            return (V_z / self.V_0 - 0.5) * self.V_width
        else:
            raise NotImplementedError(f'In KinkPotential.potential_inv_function(): unsupported potential shape {self._shape_code}')
        
    def __str__(self) -> str:
        return f'{type(self).__name__}(shape={self.shape}, V_0={self.V_0 / units.eV: <.2f}eV, ' + \
               f'width={self.V_width / units.Angstrom: <.2f}Ao)'

    def initialize_from_dictionary(self, d: dict):
        'Initialize the potential shape and parameters from a set of options organized in a dictionary'
        super().initialize_from_dictionary(d)
        if 'width' not in d:
            self.V_width = 10.0 * units.Angstrom
        else:
            d_width = d['width']
            if isinstance(d_width, float):
                self.V_width = d_width * units.Angstrom
            elif isinstance(d_width, str):
                try:
                    self.V_width = utils.float_with_units(d_width, default_units='Angstrom', 
                                                        allowed_units={'Angstrom', 'Bohr'})
                except:
                    raise ValueError('In KinkPotential.initialize_from_dictionary(): syntax error in width field')
            else:
                raise ValueError('In KinkPotential.initialize_from_dictionary(): syntax error in width field')
