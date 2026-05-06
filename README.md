# The Joint Gromov-Wasserstein Objective
## Requirements
Our code are written using Python 3.12.3 and need ChimeraX to run. Also to run our scripts you need to install the packages in `requirement.txt`.
```
pip install -r requirement.txt
```

## Data Generation
For generating the benchmarking dataset of models that we used in our experiments first go to `Data/` and download the original atomic models  as follows.
```
python download.py
```
Then partition each map into its chains using the following script.
```
python splitter.py
```
Then convert atomic models into density maps using the `mrc_generator.py` script in ChimeraX. **You Can't perform this step outside of ChimeraX environment**.

After this step the dataset is ready to use.
