# RiflesSO: *ab initio* reflectance with spin-orbit effects 
This project aims at describing reflection of charge carriers off a crystal boundary. The features planned to be implemented are:
* **various physical approximations/regimes**: quantum, quasiclassical, adiabatic, etc.;
* **fully *ab initio* treatment of the band structure**: work with as many bands as possible beyond the quadratic-dispersion or spin-degeneracy approximations;
* **support of boundary properties**: roughness, orientation, etc.;
* **import of Bloch/Wannier data** from output files of [`qimpy`](https://github.com/shankar1729/qimpy) and other DFT packages.

The code can be used either as a Python library or as a standalone command-line tool.

## Usage: Command line
```
    python -m riflesso -i task.yaml -o logfile
```
Here, the `-i` (`--input-yaml`) argument is required in order to point to the YAML-formatted input script `task.yaml` listing the options of the calculation. In contrast, the `-o` (`--output-stem`) argument, defining the log filename, is optional; by default, `RiflesSO` writes its output to the console. 

The input script contains such sections as `lattice`, `electronic-struture`, `solver`, `boundary-potential`, and `task`, which define which data to use as input, which physical approximations to use, and which quantities to caclulate and save. For a concrete example, see, e.g., [/tests/reflect_graphene.yaml].

## Usage: Python library
Individual `RiflesSO` classes can be used for various operations with the *ab initio* electronic structure pre-calculated, e.g., using [JDFTx](https://jdftx.org/):
```python
from riflesso import elecstructure
import numpy as np

# Import a Wannierized electronic structure
wannierized_estruc = elecstructure.ElectronicStructure.from_dictionary(
    { 'type':                  'Wannierized', 
      'format':                'jdftx',
      'filename.dft-output':   'jdftx_calc/graphene/scf_fine.out',
      'filename.wannier-stem': 'jdftx_calc/graphene/wannier'
    }
)

# Print a band Hamiltonian for a given k vector near the K point in BZ
k = np.array([1/3 + 0.01, 1/3, 0.0])
_, H_k = wannierized_estruc.electronic_states_BZ(k, return_value='eigenstates_Hband')
print(f'H(k = {np.array2string(k, precision=3)}) = \n{np.array2string(H_k, precision=3)}')
```

Alternatively, the whole toolchain can be initialized and run in a way equivalent to the command line:
```python
from riflesso import RiflesSO
import numpy as np

# Initialize the input parameters: lattice vectors, electronic structure, etc.
engine = RiflesSO(from_yaml = 'path/to/my.yaml')

# Calculate and write reflection coefficients to the output file, as described in my.yaml
engine.run()

# ...Do something with the engine, if necessary...
```

## Physical effects described and to be added
While light in vacuum has ***two*** `polarization` states, an electron with a given `crystal momentum` in a crystalline semiconductor can belong to ***many*** `bands`. Photon polarization can undergo transformations at the interfaces of transparent media (see, e.g., [Brewster's angle](https://en.wikipedia.org/wiki/Brewster%27s_angle), [Birefringence](https://en.wikipedia.org/wiki/Birefringence])). Reflection of electrons, due to their more complicated band dispersion in comparison with photons, can lead to interband transitions and is in no way described by the conventional reflection law in general. In contrast, a single incident electron beam (band $n$, momentum $\mathbf{k}$) gets split into a number of reflected beams with certain probabilities (or relative intensities), each of them having its own reflection angle and/or band $n'$. These relative intensities are quantified by the `reflectance`, or reflection probability $\mathcal{R}_{\mathbf{k}'n', \mathbf{k}n}$ calculated by `RiflesSO`.


## Authors, current status, ...

The project was created by [Oleg G. Kharlanov](http://theorphys.phys.msu.ru/en/staff/kharlanov.html) (Moscow State University) in collaboration with [Yuan Ping](https://directory.engr.wisc.edu/mse/Faculty/Ping_Yuan) (Wisconsin University) and [Ravishankar Sundararaman](https://mse.rpi.edu/people/faculty/ravishankar-sundararaman) (Rensselaer Polytechnic Institute). The goal is to further merge RiflesSO with [`qimpy`](https://github.com/shankar1729/qimpy) for *ab initio* quantum transport simulation of electronic/spintronic devices; however, certain methods implemented in `RiflesSO` can be useful in their own right.
**The project is currently under development.**
