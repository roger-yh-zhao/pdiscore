# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:12:16 2024

@author: roger
"""
import argparse

def unique_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='1qne', type=str, help='name')
opt = parser.parse_args()

oldnum=[]
count=0
pdb=open( "%s_pocket.pdb"%(opt.name) );pdb=pdb.readlines();chain=[i[21:26] for i in pdb if len(i.split())>8]
chain=unique_preserve_order(chain)
out=open("%s_pocket_o.pdb"%(opt.name),'w')
for j in pdb:
    if len(j.split())>8:
        oldnum.append(j[21:26])
        if len(oldnum)==1 or oldnum[-1]!=oldnum[-2]:
            count+=1
        
        j=j[:21]+ 'A %s'%( (str(count).rjust(3)) )+ j[26:]
    
    out.writelines(j)

oldnum=[]
count=0
pdb=open( "%s_d.pdb"%(opt.name) );pdb=pdb.readlines();chain=[i[21:26] for i in pdb if len(i.split())>8]
chain=unique_preserve_order(chain)
out=open("%s_d_o.pdb"%(opt.name),'w')
for j in pdb:
    if len(j.split())>8:
        oldnum.append(j[21:26])
        if len(oldnum)==1 or oldnum[-1]!=oldnum[-2]:
            count+=1
        
        j=j[:21]+ 'B %s'%( (str(count).rjust(3)) )+ j[26:]
    
    out.writelines(j)
 