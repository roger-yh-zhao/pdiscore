import prody as pr
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='1qne', type=str, help='name')
opt = parser.parse_args()

cutoff=10

pro =  pr.parsePDB("%s_p.pdb"%(opt.name), model=1)  
nuc =  pr.parsePDB("%s_d.pdb"%(opt.name), model=1)

for residue in nuc.iterResidues():
    # Change the residue name to a new value
    residue.setResname('NUC')

com = pro + nuc

ret = com.select( f'same residue as exwithin %s of resname NUC'%(cutoff) )
ret = ret.select('not resname NUC')

pr.writePDB("%s_pocket.pdb"%(opt.name), ret)
