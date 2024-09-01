import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import MDAnalysis as mda
sys.path.append(os.path.abspath(__file__).replace("test.py",".."))
from torch_geometric.loader import DataLoader
from PDIScore.data.data import VSDataset,PDBbindDataset
from PDIScore.model.utils import run_an_eval_epoch
from PDIScore.model.model import PDIScore, GraphTransformer, GatedGCN, graphGPS
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#you need to set the babel libdir first if you need to generate the pocket
os.environ["BABEL_LIBDIR"] = "/home/yihao/anaconda3/envs/gen/lib/openbabel/3.1.0"

def Input():
	p = argparse.ArgumentParser()
	p.add_argument('-of','--output_file', default='score.csv',
									help='output file of scores')
	p.add_argument('-p','--prot', default="pro.pt",
									help='Input protein graph')
	p.add_argument('-l','--lig', default="lig.pt",
									help='Input ligand graph')
	p.add_argument('-i','--ids', default="ids.npy",
									help='Input ids')
	p.add_argument('-m','--model', default="../weight/weight.pth", 
									help='trained model path')								
	p.add_argument('-e','--encoder', default="gps", choices=["gt", "gatedgcn", "gps","gpsgt"], 
									help='the feature encoders for the representation of proteins and ligands (default: "gt")')	
	p.add_argument('-o','--outprefix', default="out", 
									help='the prefix of output file (default: "out")')	
	p.add_argument('-gen_pocket','--gen_pocket', action="store_true", default=False,
									help='whether to generate the pocket')	
	p.add_argument('-c','--cutoff', default=10.0, type=float,
									help='the cutoff the define the pocket and interactions within the pocket (default: 10.0)')		
	p.add_argument('-rl','--reflig', default=None, 
									help='the reference ligand to determine the pocket(.sdf/.mol2)')								
	p.add_argument('-pl', '--parallel', default=False, action="store_true",
						help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')		
	p.add_argument('-ac', '--atom_contribution', default=False, action="store_true",
						help='whether to obtain the atom contrubution of the score.')		
	p.add_argument('-rc', '--res_contribution', default=False, action="store_true",
						help='whether to obtain the residue contrubution of the score.')
	p.add_argument('-gps_local', type=str, default="GINE", help="local_gnn_type of gps")
	p.add_argument('-gps_global', type=str, default="BigBird", help="global_model_type of gps")
	p.add_argument('-featsl', type=int, default=43, help="featsl")
	
	args = p.parse_args()
	if args.gen_pocket:
		if args.reflig is None:
			raise ValueError("if pocket is generated, the reference ligand should be provided.")
	if args.atom_contribution and args.res_contribution:
		raise ValueError("only one of the atom_contribution and res_contribution can be supported")
	return args


def scoring(prot, lig, ids, modpath,
			cut=10.0,
			gen_pocket=False,
			reflig=None,
			encoder="gt",
			atom_contribution=False,
			res_contribution=False,
			explicit_H=False, 
			use_chirality=True,
			parallel=False,
			**kwargs
			):
	"""
	prot: The input protein file ('.pdb')
	lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
	modpath: The path to store the pre-trained model
	gen_pocket: whether to generate the pocket from the protein file.
	encoder: The feature encoders for the representation of proteins and ligands.
	reflig: The reference ligand to determine the pocket.
	cut: The distance within the reference ligand to determine the pocket.
	atom_contribution: whether the decompose the score at atom level.
	res_contribution: whether the decompose the score at residue level.
	explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
	use_chirality: whether to adopt the information of chirality to represent the molecules.	
	parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
	kwargs: other arguments related with model
	"""

	data = PDBbindDataset(ids=ids,
					  ligs=lig,
					  prots=prot
					  )#;print(data.pdbids);print(data.gps);print(data.gls)

					
	test_loader = DataLoader(dataset=data[:], 
							batch_size=kwargs["batch_size"],
							shuffle=False, 
							num_workers=kwargs["num_workers"]
							)
	
	if encoder == "gt":
		ligmodel = GraphTransformer(in_channels=kwargs["num_node_featsl"], 
									edge_features=kwargs["num_edge_featsl"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
	
		protmodel = GraphTransformer(in_channels=kwargs["num_node_featsp"], 
									edge_features=kwargs["num_edge_featsp"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
	
	elif encoder == "gatedgcn":
		ligmodel = GatedGCN(in_channels=kwargs["num_node_featsl"], 
							edge_features=kwargs["num_edge_featsl"], 
							num_hidden_channels=kwargs["hidden_dim0"], 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6
							)
		
		protmodel = GatedGCN(in_channels=kwargs["num_node_featsp"], 
							edge_features=kwargs["num_edge_featsp"], 
							num_hidden_channels=kwargs["hidden_dim0"], 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6
							)
	elif encoder == "gps":
		ligmodel = graphGPS(in_channels=kwargs["num_node_featsl"], 
							edge_features=kwargs["num_edge_featsl"], 
							num_hidden_channels=kwargs["hidden_dim0"], 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6,
							local_gnn_type=kwargs["gps_local"],
							global_model_type=kwargs["gps_global"] 
							)
		
		protmodel = graphGPS(in_channels=kwargs["num_node_featsp"], 
							edge_features=kwargs["num_edge_featsp"], 
							num_hidden_channels=kwargs["hidden_dim0"], 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6,
							local_gnn_type=kwargs["gps_local"],
							global_model_type=kwargs["gps_global"] 
							)
	elif encoder == "gpsgt":
		ligmodel = graphGPS(in_channels=kwargs["num_node_featsl"], 
							edge_features=kwargs["num_edge_featsl"], 
							num_hidden_channels=kwargs["hidden_dim0"], 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6,
							local_gnn_type=kwargs["gps_local"],
							global_model_type=kwargs["gps_global"] 
							)
		
		protmodel = GraphTransformer(in_channels=kwargs["num_node_featsp"], 
									edge_features=kwargs["num_edge_featsp"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
						
	model = PDIScore(ligmodel, protmodel, 
					in_channels=kwargs["hidden_dim0"], 
					hidden_dim=kwargs["hidden_dim"], 
					n_gaussians=kwargs["n_gaussians"], 
					dropout_rate=kwargs["dropout_rate"], 
					dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])
	
	checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
	model.load_state_dict(checkpoint['model_state_dict']) 
	if atom_contribution:
		preds, at_contrs, _ = run_an_eval_epoch(model, 
												test_loader, 
												pred=True, 
												atom_contribution=True, 
												res_contribution=False, 
												dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
		
		atids = ["%s%s"%(a.GetSymbol(),a.GetIdx()) for a in data.ligs[0].GetAtoms()]
		return data.ids, preds, atids, at_contrs
	
	elif res_contribution:
		preds, _, res_contrs = run_an_eval_epoch(model, 
												test_loader, 
												pred=True, 
												atom_contribution=False, 
												res_contribution=True, 
												dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
		u = mda.Universe(data.prot)
		resids = ["%s_%s%s"%(x[0],y,z) for x,y,z in zip(u.residues.chainIDs, u.residues.resnames, u.residues.resids)]
		return data.ids, preds, resids, res_contrs
	else:	
		preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
		return data.pdbids, preds


def main():
	inargs = Input()
	args={}
	args["batch_size"] = 100
	args["dist_threhold"] = 5.
	args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
	args["num_workers"] = 10
	args["num_node_featsp"] = 41
	args["num_node_featsl"] = inargs.featsl
	args["num_edge_featsp"] = 5
	args["num_edge_featsl"] = 5
	args["hidden_dim0"] = 128
	args["hidden_dim"] = 128 
	args["n_gaussians"] = 10
	args["dropout_rate"] = 0.15
	args["output_file"] = inargs.output_file
	args["gps_local"] = inargs.gps_local
	args["gps_global"] = inargs.gps_global
	if inargs.atom_contribution:
		ids, scores, atids, at_contrs = scoring(prot=inargs.prot, 
											lig=inargs.lig, 
											ids=inargs.ids,
											modpath=inargs.model,
											cut=inargs.cutoff,
											gen_pocket=inargs.gen_pocket,
											reflig=inargs.reflig,
											encoder=inargs.encoder,
											atom_contribution=True,
											explicit_H=False, 
											use_chirality=True,
											parallel=inargs.parallel,
											**args
											)
		df = pd.DataFrame(at_contrs).T
		df.columns= ids
		df.index = atids
		df = df[df.apply(np.sum,axis=1)!=0].T
		dfx = pd.DataFrame(zip(*(ids, scores)),columns=["id","score"])
		dfx.index = dfx.id
		df = pd.concat([dfx["score"],df],axis=1)
		df.sort_values("score", ascending=False, inplace=True)	
		df.to_csv("%s_at.csv"%inargs.outprefix)
	elif inargs.res_contribution:
		ids, scores, resids, res_contrs = scoring(prot=inargs.prot, 
											lig=inargs.lig, 
											ids=inargs.ids,
											modpath=inargs.model,
											cut=inargs.cutoff,
											gen_pocket=inargs.gen_pocket,
											reflig=inargs.reflig,
											encoder=inargs.encoder,
											res_contribution=True,
											explicit_H=False, 
											use_chirality=True,
											parallel=inargs.parallel,
											**args
											)
		df = pd.DataFrame(res_contrs).T
		df.columns= ids
		df.index = resids
		df = df[df.apply(np.sum,axis=1)!=0].T
		dfx = pd.DataFrame(zip(*(ids, scores)),columns=["id","score"])
		dfx.index = dfx.id
		df = pd.concat([dfx["score"],df],axis=1)
		df.sort_values("score", ascending=False, inplace=True)	
		df.to_csv("%s_res.csv"%inargs.outprefix)			
	else:
		ids, scores = scoring(prot=inargs.prot, 
							lig=inargs.lig, 
							ids=inargs.ids,
							modpath=inargs.model,
							cut=inargs.cutoff,
							gen_pocket=inargs.gen_pocket,
							encoder=inargs.encoder,
							reflig=inargs.reflig,
							explicit_H=False, 
							use_chirality=True,
							parallel=inargs.parallel,
							**args
							)
		df = pd.DataFrame(zip(*(ids, scores)),columns=["id","score"])
		#df.sort_values("score", ascending=False, inplace=True)
		#df.to_csv("%s.csv"%inargs.outprefix, index=False)

		f_result = open( args["output_file"], 'w')
		ids = np.load(inargs.ids)    
		for i in range(len(scores)):
		    f_result.writelines("%s,%s\n"%(ids[0][i], scores[i])) 


if __name__ == '__main__':
    main()
