
# Assignment *number 1*

- name: Joshua McPherson
- student ID: 20687868

## Dependencies
- numpy
-matplotlib

## Running `main.py`

To run `main.py`, use:

```sh
python main.py data/x.txt 
```

where x is the name of the input file, results will be stored in outputs/x.txt

to run `main.py` in verbose mode, use:

```sh
python main.py data/x.txt -v
```

This will display a matplotlib graph of the KL Divergence between the Hopfield network 
and the data over the epochs of training. Note however hopfield networks with Hebbian Learning 
do not benefit from additional training epochs so this will always be constant.