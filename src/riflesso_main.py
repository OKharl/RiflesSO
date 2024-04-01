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
from .utils import timeit, normalize, nparr2str, plane_vectors, write_array_binary, read_array_binary
from .elecstructure import ElectronicStructure, BlochState
from .boundarypot import BoundaryPotential, KinkPotential
from .projector import KgridProjector, Kgrid

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
    
    riflesso_version = '0.1'                        # The version displayed on "riflesso --version"

    arg_parser:             ArgumentParser          # Parser of command-line arguments

    # Parts of the toolchain
    lattice:                CrystallineHalfSpace
    electronic_structure:   Type[ElectronicStructure]
    boundary_potential:     Type[BoundaryPotential]
    reflection_solver:      Type['ReflectionSolver']
    kgrid_projector:        Type['KgridProjector']

    # Logging stream and filename
    log_stream:             IO
    log_filename:           str

    # Type of task to be done
    task_type:              str

    # Boundary normals, for which the calculations are to be performed
    normals:                List[Union[List[int], NDArray[np.float]]]
    kpoint_grid:            Type[Kgrid]

    def __init__(self, *, 
                 from_command_line: Optional[str] = 'do_not_use',
                 from_dict: Optional[dict] = None,
                 from_yaml: Optional[Union[str, yaml.YAMLObject, dict]] = None
                ):
        '''Initialize a RiflesSO engine form a command line, an input YAML-formatted script, or a dictionary.
        To use the command line of the process (i.e., `argv`), set from_command_line = None or 'from_os'.
        As for YAML scripts, `from_yaml` can be both a parsed YAML (i.e., a dict), or a YAML filename.
        '''
        # Initialize defaults
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
        self.normals = [[1, 0, 0]]    # Just a single normal defined via Miller indices

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
            else:
                raise ValueError('In RiflesSO.__init__(): unsupported type of from_yaml parameter')
            if 'riflesso' not in yaml_dict:
                raise ValueError('In RiflesSO.__init__(): section "riflesso" not found in YAML file')
            self.initialize_from_dictionary(yaml_dict['riflesso'])
        elif from_command_line != 'do_not_use':
            if from_command_line == 'from_os' or from_command_line == None:
                self.initialize_from_command_line()
            else:
                self.initialize_from_command_line(from_command_line)
            

    def __del__(self):
        "Destructor of RiflesSO engine: closes the log streams"
        if self.log_filename != 'console' and self.log_stream is not None:
            self.log_stream.close()

    def run(self):
        pass

    def set_logfile(self, filename: Optional[str] = 'console'):
        "Set log filename, closing the previous used one if necessary"
        if self.log_filename == filename:
            return
        if self.log_filename != 'console':
            self.log_stream.close()
        self.log_filename = filename
        self.log_stream = None if filename == 'console' else open(filename, 'wt')

    def read_input_yaml_script(self, filename: str):
        "Input a YAML-formatted input file and initialize the parameters of RiflesSO"
        with open(filename, 'rt') as f:
            input_script = yaml.safe_load(f)
            if 'riflesso' not in input_script:
                raise ValueError('In RiflesSO.read_input_yaml_script(): section "riflesso" not found in YAML file')
            self.initialize_from_dictionary(input_script['riflesso'])

    def initialize_from_command_line(self, command_line: Optional[str] = None):
        '''Parse command_line and initialize RiflesSO engine from the arguments in it. 
        If `command_line` is None, use the system command line of the process instead.
        '''
        p = self.arg_parser
        if command_line is None:
            args = p.parse_args()  # Use OS's command line instead
        else:
            args = p.parse_args(command_line.strip())
        if args.version:
            print(str(self.riflesso_version))
            return
        if args.output_stem:
            self.set_logfile(args.output_stem + '.log')
        if args.input_yaml:
            self.read_input_yaml_script(args.input_yaml)
    
    def log(self, msg: str, *args, **kwargs):
        "Write a message to the log file or to console if the log file is not set up"
        if self.log_filename == 'console':
            print(msg, *args, **kwargs)
        elif self.log_filename and self.log_stream is not None:
            print(msg, file=self.log_stream, *args, **kwargs)

    def initialize_from_dictionary(self, d: dict):
        "Initialize RiflesSO engine from a dictionary (a `riflesso` section of the YAML input script)"
        if 'task' not in d:
            raise ValueError('In RiflesSO.initialize_from_dictionary(): section "task" missing')
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
        if 'task' in d:
            d_task = dict(d['task'])
            if 'boundary-normals' in d_task:
                self.normals = self.generate_normals_from_dictionary(d_task['boundary-normals'])
                self.log(f'{len(self.normals)} boundary normal vectors generated/loaded')
            self.task_type = d_task.get('task.type', 'reflection-coeffs')
        # TODO: parse other keywords


    def create_argument_parser(self):
        "Create a parser object for potential scanning of the command line passed to RiflesSO"
        self.arg_parser = ArgumentParser(prog='RiflesSO', 
                                         description='ab initio reflectance solver for crystalline materials')
        self.arg_parser.add_argument('-i', '--input-yaml', type=str, 
                                     help='Input YAML filename describing the tasks to be done', required=False)
        self.arg_parser.add_argument('-o', '--output-stem', type=str,
                                     help='Stem for output files, e.g., those with the calculated reflectances', required=False)
        self.arg_parser.add_argument('-v', '--version', action='store_true', help='Display the version information', required=False)

    def generate_normals_from_dictionary(self, d: dict) -> List[Union[NDArray[np.float], List[int]]]:
        '''Generate a set of normal vectors (np.array[float]) or Miller indices (list[int])
        from a section of a configuration dictionary.
        '''
        t = d.get('type', 'from-file')
        if t == 'from-file':
            if 'filename' not in d:
                raise ValueError('In RiflesSO.generate_normals_from_dictionary(): normal file name missing')
            fn = d['filename']
            fmt = d.get('format', 'table.Cartesian')
            if fmt not in {'table.Cartesian', 'table.Miller', 'binary.Cartesian'}:
                raise ValueError('In RiflesSO.generate_normals_from_dictionary(): unknown normal file format')
            if fmt == 'table.Cartesian':
                normals = [normalize(n) for n in np.loadtxt(fn, dtype=float)]
            elif fmt == 'table.Miller':
                normals = [list(n) for n in np.loadtxt(fn, dtype=int)]
            elif fmt == 'binary.Cartesian':
                normals = np.load(fn, dtype=np.float64)
                normals = [normalize(n) for n in normals.reshape(normals.size // 3, 3)]  # Assuming dim = 3 
            return normals
        elif t == 'list':
            if 'normal-vectors.Cartesian' in d:
                normals = [normalize(n) for n in np.array(d['normal-vectors.Cartesian'], dtype=float)]
            elif 'normal-vectors.Miller' in d:
                normals = [list(n) for n in np.array(d['normal-vectors.Cartesian'], dtype=int)]
            else:
                raise ValueError('In RiflesSO.generate_normals_from_dictionary(): "normal-vectors" list not found')
            return normals
        elif t == 'in-plane':
            if 'plane-normal' in d:
                a, b = plane_vectors(np.array(d['plane-normal'], dtype=float))
            elif 'plane-a.Cartesian' in d and 'plane-b.Cartesian' in d:
                a, b = [np.array(d['plane-' + v + '.Cartesian'], dtype=float) for v in ['a', 'b']]
                a, b = plane_vectors(np.cross(a, b))
            else:
                raise ValueError('In RiflesSO.generate_normals_from_dictionary(): for type=in-plane, ' +
                                 'the plane containing normals should be specified')
            phi_range = d.get('degrees-range', [0, 360, 10])
            phi_range = np.arange(*phi_range, dtype=float) * units.Degree
            normals = [a * np.cos(phi) + b * np.sin(phi) for phi in phi_range]
        else:
            raise ValueError('In RiflesSO.generate_normals_from_dictionary(): unknown normals generation type')
        return normals

    def generate_task_kpoints_from_dictionary(self, k_point_grid_clause: Union[List[int], str, dict]) -> Sequence:
        '''Generate a Kgrid object or just a list of fractional k-point coordinates from a `k-point-grid` clause 
        of the input script. The result should be iterable
        '''
        pass
    
    def write_reflection_coeffs_file(self, 
                                     rho_data: List[List[Tuple[NDArray[np.float], NDArray[np.complex]]]],
                                     filename: str, format: str ='binary'):
        '''Write reflection coefficients data to an output binary or text file
        '''
        if format == 'binary':
            with open(filename, 'wb') as f:
                # 0. Write the header
                f.write(b'Reflection coefficients file created by OK RiflesSO')
                # 1. Write the lattice vectors, the k points [do not write the Bloch wave functions so far]
                write_array_binary(self.lattice.lattice_vectors, f, dtype=np.float64)
                kpts = np.fromiter(self.kpoint_grid, dtype=np.float64)
                #nbands = len(rho_data) // kpts.shape[0]
                nbands = self.electronic_structure.nb
                write_array_binary(kpts, f, dtype=np.float64)
                # 2. For each k and n, write the reflected states and their complex amplitudes
                for ikn, rho_k_n in enumerate(rho_data):
                    ik, n = ikn // nbands, ikn % nbands
                    # TODO: optimize writing of the reflected wave functions to avoid reordering of the bands
                    for k_refl, u_refl in rho_k_n:
                        write_array_binary(k_refl, f, dtype=np.float64)
                        write_array_binary(u_refl, f, dtype=np.complex128)
        else:
            raise ValueError('In RiflesSO.write_reflection_coeffs_file(): unknown output file format')



