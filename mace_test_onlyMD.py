from mace.calculators import mace_mp
from ase import build

from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase import Atoms
from ase.build import bulk
from ase.visualize import view
import numpy as np
from ase.build import add_vacuum
from ase.optimize import LBFGS, BFGS
from ase import Atom

from ase.io.trajectory import Trajectory
from ase.md import Langevin, Bussi
from ase import io





macemp = mace_mp() # return ASE calculator

m_g_relaxed = io.read('structures/m_g_relaxed.xyz')


#--------------------------------MD--------------------------------

def printenergy(a=m_g_relaxed):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    
def remove_com_motion(atoms, directions=[True, True, False]):
    """
    Remove center-of-mass motion in specified directions.
    directions=[True, True, False] removes motion in x and y but keeps z motion.
    """
    momenta = atoms.get_momenta()
    mass = atoms.get_masses()
    com_momentum = np.sum(momenta.T * mass, axis=1) / np.sum(mass)

    for i, remove in enumerate(directions):
        if remove:
            momenta[:, i] -= com_momentum[i]  # Remove COM motion in x or y

    atoms.set_momenta(momenta)
    
# m_g
m_g_relaxed.calc = macemp
T_init = 300  # Initial temperature in K
MaxwellBoltzmannDistribution(m_g_relaxed, temperature_K=T_init)
remove_com_motion(m_g_relaxed)  # Remove drift in x and y

#dyn = Langevin(m_g_relaxed, 1 * units.fs, T_init * units.kB, 0.01)
dyn = Bussi(m_g_relaxed, 0.5 * units.fs, T_init, 0.01)
#n_steps = 5000 + 20000 # 5ps pre-equilibration + 20ps production

n_steps = 5000

dyn.attach(printenergy, interval = 100) 

traj = Trajectory('mg_Bussi_300K_5000step.traj', 'w', m_g_relaxed)
dyn.attach(traj.write, interval = 10)
printenergy()
dyn.run(n_steps)