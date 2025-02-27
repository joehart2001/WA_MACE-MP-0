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


#macemp = mace_mp(dispersion=True, default_dtype="float64"
#macemp_omat = mace_mp(model="/Users/joehart/Desktop/0_Cambridge/0_MPhil_Scientific_Computing/Written_assignments/MACE-MP-0/Notebooks_mace/mace_test/mace-omat-0-medium.model", dispersion=True, default_dtype="float64")
macemp_omat = mace_mp(model="/home/raid/jh2536/WA_macemp0/mace-omat-0-medium.model", dispersion=True, default_dtype="float64", device='cuda', enable_cueq=True)



m_goh_relaxed = io.read("/home/raid/jh2536/WA_macemp0/structures/m_goh_relaxed_flip_omat.xyz")




#--------------------------------MD--------------------------------

import time
start_time = time.time()

def printenergy(a=m_goh_relaxed):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    elapsed_time = time.time() - start_time  # Time in seconds
    elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV Time Elapsed: %dm %.1fs' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, int(elapsed_min), int(elapsed_sec)))


from ase.constraints import FixSubsetCom

    
# m_g
#m_g_relaxed.calc = macemp
m_goh_relaxed.calc = macemp_omat
T_init = 300  # Initial temperature in K
MaxwellBoltzmannDistribution(m_goh_relaxed, temperature_K=T_init)


#dyn = Langevin(m_g_relaxed, 1 * units.fs, T_init * units.kB, 0.01)
dyn = Bussi(m_goh_relaxed, 1 * units.fs, T_init, 0.01)
#n_steps = 5000 + 20000 # 5ps pre-equilibration + 20ps production
n_steps = 25000

dyn.attach(printenergy, interval = 1000) 
#dyn.attach(lambda: remove_COM_momentum(m_g_relaxed), interval=1)

#m_g_relaxed.set_constraint(FixCom())

tags = m_goh_relaxed.get_tags()

mxene_indices = np.where(np.isin(tags, [0, 1, 2, 3]))[0]  # MXene atoms
graphene_indices = np.where(np.isin(tags, [10, 11, 12]))[0]  # Graphene atoms

fix_com_mxene = FixSubsetCom(indices=mxene_indices)
fix_com_graphene = FixSubsetCom(indices=graphene_indices)
m_goh_relaxed.set_constraint([fix_com_mxene, fix_com_graphene])

#m_g_relaxed.set_constraint(FixSubsetCom(mxene_indices))
#m_g_relaxed.set_constraint(FixSubsetCom(graphene_indices))


#traj = Trajectory('mg_test.traj', 'w', m_g_relaxed)
traj = Trajectory('/home/raid/jh2536/WA_macemp0/traj/mgoh_omat_flipped_25000.traj', 'w', m_goh_relaxed)
dyn.attach(traj.write, interval = 10)
printenergy()
dyn.run(n_steps)