Build environment by conda:
````
conda env create -f environment.yml
````


Generate pocket:
````
python pocket.py --name 1qne
````

Reorder the residue/nucleotide of protein/nucleic acid
````
python reorder.py --name 1qne
````

Convert pdb structure to graph
````
python pdb2graph.py -idf ids.txt
````

Output the final score
````
python test.py -of score.csv
````
