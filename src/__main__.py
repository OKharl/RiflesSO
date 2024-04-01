# =================================================================================================
# __main__.py: the RiflesSO entry point when called from command line
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================

from typing import Optional, Sequence, Union, Type, IO
from numpy.typing import NDArray
import numpy as np

from .lattice import CrystalLattice, CrystallineHalfSpace
from .elecstructure import ElectronicStructure, BlochState
from .boundarypot import BoundaryPotential, KinkPotential
from .projector import KgridProjector
from .riflesso_main import ReflectionTask, RiflesSO

riflesso = RiflesSO(from_command_line='from_os')
riflesso.run()
