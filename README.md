# RiflesSO: *ab initio* electron reflectance with spin-orbit effects 
This project aims at describing reflection of charge carriers off a crystal boundary. The features planned to be implemented are:
* **various physical approximations/regimes**: quantum, quasiclassical, adiabatic, etc.;
* **fully *ab initio* treatment of the band structure**: work with as many bands as possible beyond the quadratic-dispersion or spin-degeneracy approximations;
* **support of boundary properties**: roughness, orientation, etc.;
* **import of Bloch/Wannier data** from output files of [`qimpy`](https://github.com/shankar1729/qimpy) and other DFT packages.

The code is designed to be used either as a Python library or as a standalone command-line tool. For further details of the physics, the usage modes, and the features, see a [**description**](/doc/riflesso.pdf).


<!--
## Usage: Python library
...

## Usage: Command line
...
-->

## Physical effects described and to be added
While light in vacuum has ***two*** polarization states, an electron with a given crystal momentum in a crystalline semiconductor can belong to ***many*** bands. Photon polarization can undergo transformations at the interfaces of transparent media (see, e.g., [Brewster's angle](https://en.wikipedia.org/wiki/Brewster%27s_angle), [Birefringence](https://en.wikipedia.org/wiki/Birefringence])). Reflection of electrons, due to their more complicated band dispersion in comparison with photons, can lead to interband transitions and is in no way described by the conventional reflection law in general. In contrast, a single incident electron beam (band $n$, momentum $\mathbf{k}$) gets split into a number of reflected beams with certain probabilities (or relative intensities), each of them having its own reflection angle and/or band $n'$. These relative intensities are quantified by the reflectances, or reflection probabilities $\mathcal{R}(\mathbf{k}n \to \mathbf{k}'n')$ calculated by `RiflesSO`.

## Authors, current status, ...

The project is being developed created by [Oleg G. Kharlanov](http://theorphys.phys.msu.ru/en/staff/kharlanov.html) (Moscow State University) in contact with [Yuan Ping](https://directory.engr.wisc.edu/mse/Faculty/Ping_Yuan) (Wisconsin University) and [Ravishankar Sundararaman](https://mse.rpi.edu/people/faculty/ravishankar-sundararaman) (Rensselaer Polytechnic Institute). The goal was to further merge RiflesSO with [`qimpy`](https://github.com/shankar1729/qimpy) for *ab initio* quantum transport simulation of electronic/spintronic devices; however, certain methods implemented in `RiflesSO` can be useful in their own right.

**The project is currently under development.**
