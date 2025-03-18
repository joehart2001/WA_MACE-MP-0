import numpy as np


# more complicated as first seems -> need to make a robust function to handle:
#   - Two O layers (close and far from graphene)
#   - C layer in MXene and graphene
#   - when we functionalise graphene with O groups, we can't use the O layer avg as it includes the O or OH groups
# use tags to identify atoms


# Tags
# 0: mxene Ti
# 1: bottom mxene O
# 2: mxene carbon
# 3: top mxene O
# 10: graphene C
# 11: graphene O
# 12: graphene H





def calculate_interlayer_spacing(atoms, avg_z=False, bottom=False):
    # define interlayer spacing as the distance between avg C z positions in graphene and 
    # the avg O z postions in mxene's O layer closest to graphene
    
    #graphene_C_atoms = [atom.position[2] for atom in atoms if atom.symbol == 'C' and graphene_C(atom, atoms)]
    graphene_C_atoms = [atom.position[2] for atom in atoms if atom.tag == 10]

    # Get MXene oxygens (excluding functional groups)
    if bottom == True:
        mxene_O_atoms = [atom.position[2] for atom in atoms if atom.tag == 1]
    else:
        #mxene_O_atoms = [atom.position[2] for atom in atoms if atom.symbol == 'O' and top_mxene_O(atom, atoms, bottom=bottom)]
        mxene_O_atoms = [atom.position[2] for atom in atoms if atom.tag == 3]


    # Compute average z-positions
    avg_z_C = np.mean(graphene_C_atoms)
    avg_z_O = np.mean(mxene_O_atoms)
    
    if avg_z == True:
        return avg_z_C - avg_z_O, avg_z_C, avg_z_O

    # Compute interlayer spacing (MXene O above Graphene C)
    return avg_z_C - avg_z_O


def set_interlayer_spacing(heterostructure, height):
    heterostructure = heterostructure.copy()
    
    current_spacing, _, avg_z_O = calculate_interlayer_spacing(heterostructure, avg_z=True)
    
    difference = height - current_spacing
    
    # shift every atom above mxene's top O layer (to include functional groups of graphene if present)
    graphene_C_O_H_atoms = [atom for atom in heterostructure if atom.tag in [10, 11, 12]]
    #graphene_C_O_H_atoms = [atom for atom in heterostructure if atom.position[2] > (avg_z_O + 0.2)]
    
    for atom in graphene_C_O_H_atoms:
        atom.position[2] += difference
    
    return heterostructure