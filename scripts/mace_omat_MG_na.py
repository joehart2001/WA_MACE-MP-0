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


macemp_omat = mace_mp(model="/home/jh2536/rds/hpc-work/mace_data/models/mace-omat-0-medium.model", dispersion=True, default_dtype="float64", device='cuda', enable_cueq=True)

#mxene_relaxed = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/relaxed_mxene.xyz")
#graphene_atoms = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/graphene_atoms.xyz")
#go_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/go_omat_relax.xyz")
#goh_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/goh_omat_relax.xyz")

m_g_na_relaxed_omat = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/m_g_na_relaxed_omat.xyz")




#--------------------------------MD--------------------------------

import time
start_time = time.time()

def printenergy(a=m_g_na_relaxed_omat):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    elapsed_time = time.time() - start_time
    elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV Time Elapsed: %dm %.1fs' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, int(elapsed_min), int(elapsed_sec)))


from ase.constraints import FixSubsetCom

    
m_g_na_relaxed_omat.calc = macemp_omat
T_init = 300
MaxwellBoltzmannDistribution(m_g_na_relaxed_omat, temperature_K=T_init)


dyn = Bussi(m_g_na_relaxed_omat, 1 * units.fs, T_init, 0.01)
n_steps = 100000

dyn.attach(printenergy, interval = 1000) 

tags = m_g_na_relaxed_omat.get_tags()

mxene_indices = np.where(np.isin(tags, [0, 1, 2, 3]))[0]  # MXene atoms
graphene_indices = np.where(np.isin(tags, [10, 11, 12]))[0]  # Graphene atoms

fix_com_mxene = FixSubsetCom(indices=mxene_indices)
fix_com_graphene = FixSubsetCom(indices=graphene_indices)
m_g_na_relaxed_omat.set_constraint([fix_com_mxene, fix_com_graphene])


traj = Trajectory('/home/jh2536/rds/hpc-work/mace_data/traj/mg_na_omat_100000traj', 'w', m_g_na_relaxed_omat)
dyn.attach(traj.write, interval = 10)
printenergy()
dyn.run(n_steps)
