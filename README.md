PDIScore
===
![Uploading 1.png…]()

The paper is accessible at https://www.nature.com/articles/s41401-025-01688-3

Build environment
-------
````
conda create --prefix xxx --file ./requirements_conda.txt      
pip install -r ./requirements_pip.txt
````

Example
-------
Generate pocket:
````
cd test
python pocket.py --name 1qne
````

Reorder the residue/nucleotide of protein/nucleic acid:
````
python reorder.py --name 1qne
````

Convert pdb structure to graph:
````
python pdb2graph.py -idf ids.txt
````

Output the final score:
````
python test.py -of score.csv
````
