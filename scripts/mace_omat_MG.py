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

# mxene_relaxed = io.read("relaxed_mxene.xyz")
# graphene_atoms = io.read("graphene_atoms.xyz")
# go_relax = io.read("go_relax.xyz")
# goh_relax = io.read("goh_relax.xyz")
mxene_relaxed = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/relaxed_mxene.xyz")
graphene_atoms = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/graphene_atoms.xyz")
go_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/go_omat_relax.xyz")
goh_relax = io.read("/home/jh2536/rds/hpc-work/mace_data/structures/goh_omat_relax.xyz")







#--------------------------------Create heterostructures--------------------------------


def create_heterostructure(mxene, graphene_layer, height):
    # Stack MXene and graphene-based layer with a given interlayer spacing (height)
    graphene_layer = graphene_layer.copy()
    
    mxene_avg_top_O_z = np.mean([atom.position[2] for atom in mxene if atom.tag == 3])

    graphene_layer.positions[:, 2] += mxene_avg_top_O_z - np.average(graphene_layer.positions[:, 2]) + height
    heterostructure = mxene + graphene_layer
    print("MXene avg top O z: ", mxene_avg_top_O_z)

    return heterostructure


m_g = create_heterostructure(mxene_relaxed, graphene_atoms, 4.5)
m_go = create_heterostructure(mxene_relaxed, go_relax, 4.5)
m_goh = create_heterostructure(mxene_relaxed, goh_relax, 4.5)





#--------------------------------Geometry optimisation--------------------------------

m_g_relaxed = m_g.copy()
m_g_relaxed.calc = macemp_omat
optimiser = BFGS(m_g_relaxed)
optimiser.run(fmax=0.001, steps=5000)

# m_go_relaxed = m_go.copy()
# m_go_relaxed.calc = mace_mp()
# optimiser = BFGS(m_go_relaxed)
# optimiser.run(fmax=0.001, steps=5000)

# m_goh_relaxed = m_goh.copy()
# m_goh_relaxed.calc = mace_mp()
# optimiser = BFGS(m_goh_relaxed)
# optimiser.run(fmax=0.001, steps=5000)




#--------------------------------MD--------------------------------

import time
start_time = time.time()

def printenergy(a=m_g_relaxed):
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    elapsed_time = time.time() - start_time
    elapsed_min, elapsed_sec = divmod(elapsed_time, 60)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV Time Elapsed: %dm %.1fs' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, int(elapsed_min), int(elapsed_sec)))


from ase.constraints import FixSubsetCom

    
# m_g
m_g_relaxed.calc = macemp_omat
T_init = 300
MaxwellBoltzmannDistribution(m_g_relaxed, temperature_K=T_init)


#dyn = Langevin(m_g_relaxed, 1 * units.fs, T_init * units.kB, 0.01)
dyn = Bussi(m_g_relaxed, 1 * units.fs, T_init, 0.01)
n_steps = 25000 # 5ps pre-equilibration + 20ps production

dyn.attach(printenergy, interval = 1000) 

tags = m_g_relaxed.get_tags()

mxene_indices = np.where(np.isin(tags, [0, 1, 2, 3]))[0]  # MXene atoms
graphene_indices = np.where(np.isin(tags, [10, 11, 12]))[0]  # Graphene atoms

fix_com_mxene = FixSubsetCom(indices=mxene_indices)
fix_com_graphene = FixSubsetCom(indices=graphene_indices)
m_g_relaxed.set_constraint([fix_com_mxene, fix_com_graphene])


traj = Trajectory('/home/jh2536/rds/hpc-work/mace_data/traj/mg_omat_25000.traj', 'w', m_g_relaxed)
dyn.attach(traj.write, interval = 10)
printenergy()
dyn.run(n_steps)
