# =================================================================================================
# Units.py: units and physical constants [in CGS] for RiflesSO
# -------------------------------------------------------------------------------------------------
# A part of RiflesSO project created by Oleg Kharlanov (O.K.), 2024, 
# for further merging with R.Sundararaman's qimpy project.
# =================================================================================================


Pi = 3.1415926535897932384626433832795
Degree = Pi / 180                      

# CGS units and their aliases
cm   = 1.0
sec  = 1.0
gram = 1.0
erg  = gram * (cm / sec) ** 2
statCoulomb = 1.0
Gauss = 1.0
FineStructureConstant = 1.0 / 137.035999084
ElectronVolt = 1.602176634e-12 * erg
eV = ElectronVolt
meV = 0.001 * eV
MeV = 1.0e6 * eV
FermiConstant = 1.166e-11 / (MeV*MeV)
ProtonRestEnergy = 938.27208816 * MeV
NeutronRestEnergy = 939.56542052 * MeV
ElectronRestEnergy = 0.51099895000 * MeV
PlanckConstantReduced = 1.054571817e-27 * erg * sec
SpeedOfLight = 2.99792458e+10 * cm / sec
ElectronMass = ElectronRestEnergy / SpeedOfLight**2
AvogadroConstant = 6.02214076e+23    # mol^(-1)
BohrMagneton = 9.27402e-21 * (statCoulomb * cm)
SchwingerField = 4.414e+13 * Gauss
ElementaryCharge = (PlanckConstantReduced * SpeedOfLight * FineStructureConstant) ** 0.5
BohrRadius = PlanckConstantReduced**2 / (ElectronMass * ElementaryCharge**2)

# some SI and derived units
meter = 100 * cm
kg = 1000 * gram
Angstrom = 1e-8 * cm
fs = 1e-15 * sec;
ps = 1e-12 * sec;
calorie = 4.184e+7 * erg
kcal = 1000.0 * calorie
dyne = erg / cm;
Joule = kg * (meter/sec) ** 2;
Newton = Joule / meter;
Hartree = FineStructureConstant**2 * ElectronRestEnergy;
amu = 1.6605e-24 * gram;
inverseCentimeter = (2 * Pi * SpeedOfLight) / cm;   # Gives the omega value

# Aliases
GF = FermiConstant
hbar = PlanckConstantReduced
c = SpeedOfLight
NA = AvogadroConstant
muB = BohrMagneton
H0 = SchwingerField
km = 1e5 * cm
m_e = ElectronMass
e_0 = ElementaryCharge
alpha = FineStructureConstant
r_B = BohrRadius

if __name__ == '__main__':
    print(f'e_0 = {e_0} abs.units');
    print(f'Ha  = {Hartree / eV} eV');
    print(f'r_B = {r_B / Angstrom} Angstrom');
