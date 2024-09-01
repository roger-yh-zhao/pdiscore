import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
sys.path.append(os.path.abspath(__file__).replace("pdb2graph.py",".."))
from torch_geometric.loader import DataLoader
from PDIScore.data.data import PDB_convert
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#you need to set the babel libdir first if you need to generate the pocket
os.environ["BABEL_LIBDIR"] = "/home/yihao/anaconda3/envs/rtm/lib/openbabel/3.1.0"

def Input():
	p = argparse.ArgumentParser()

	p.add_argument('-idf','--ids', default='ids.txt',
									help='file of tested pdb id')								
	p.add_argument('-pl', '--parallel', default=False, action="store_true",
						help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')		
	args = p.parse_args()
	return args
        

###################################################################################################################################################

def main():
	inargs = Input()
	args={}

	inp = open(inargs.ids);inp = inp.readlines();ids = [i[:-1] for i in inp]

	ligslist = ["lig.pt"] + ["%s_d_o.pdb"%(id) for id in ids]
	proslist = ["pro.pt"] + ["%s_pocket_o.pdb"%(id) for id in ids]

	data = PDB_convert(ligs=ligslist,
    				pros=proslist,
    				parallel=inargs.parallel);

	np.save("ids.npy", [ [ id for id in ids  ], [ i for i in range(len(ids)) ] ])

if __name__ == '__main__':
    main()

