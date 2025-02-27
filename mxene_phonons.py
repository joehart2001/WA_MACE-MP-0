from ase.phonons import Phonons
from ase.dft.kpoints import bandpath
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ase import io
from mace.calculators import mace_mp
from ase import build
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import numpy as np

from ase.parallel import world


mxene_relaxed = io.read("relaxed_mxene.xyz")


special_points = {
    'Γ': [0., 0., 0.],
    'H': [0.5, -0.5, 0.5],
    'N': [0., 0., 0.5],
    'P': [0.25, 0.25, 0.25],
}
npoints = 100   # Number of points in the band structure
delta = 0.05

path_segments = ['Γ', 'H', 'N', 'Γ', 'P', 'H', 'P', 'N']
path_labels = ['Γ', 'H', 'N', 'Γ', 'P', 'H', 'P', 'N']


mxene_relaxed.calc = mace_mp()
ph = Phonons(mxene_relaxed, mace_mp(), supercell=(1, 1, 1), delta=delta)
ph.clean()  # Clean previous results to avoid caching

ph.run()

# Ensure all processes are done before proceeding
world.barrier()

if world.rank == 0:
    ph.read(acoustic=True)
    path = bandpath(path_segments, mxene_relaxed.cell, npoints=npoints, special_points=special_points)
    bands = ph.get_band_structure(path)
    frequencies = bands.energies.T

    # Save the phonon band structure
    np.save("mxene_phonon_frequencies.npy", frequencies)
    print("Saved phonon frequencies to mxene_phonon_frequencies.npy")

# # Define the path through the Brillouin zone
# path = bandpath(path_segments, mxene_relaxed.cell, npoints=npoints, special_points=special_points)
# bands = ph.get_band_structure(path)
# frequencies = bands.energies.T


# # save the phonon band structure
# np.save("mxene_phonon_frequencies.npy", frequencies)
# print("Saved phonon frequencies to phonon_frequencies.npy")