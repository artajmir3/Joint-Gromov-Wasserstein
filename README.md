# The Joint Gromov-Wasserstein Objective
This repository is the implementation of in the paper "The Joint Gromov Wasserstein Objective for Multiple Object Matching".
If you use this work for your research, please cite the paper:
```
@article{riahi2025joint,
  title={The Joint Gromov Wasserstein Objective for Multiple Object Matching},
  author={Tajmir Riahi, Aryan and Dao Duc, Khanh},
  journal={arXiv preprint arXiv:2511.16868},
  year={2025}
}
```

## Requirements
This pacckage is installable via `pip`. It solely relies on `numpy` and `POT` packages, however, to run the experiments a few other packages are needed as well. Also to generate the benchmarking data our scripts require ChimeraX to run. To install the dependencies and the package, run the following command on your terminal:

```
pip install -r requirement.txt
pip install jgw
```

## Simple Usage
For a simple usage of this package first import the essentials.
```
import numpy as np
from jgw import jgw_wrapper, jgw_solver, generate_d
```

Here is a toy example, `points_x` and `points_y` need to be in the format of a list of `np` arrays each one for one cluster.
```
points_x = [np.array([[0, 0, 0]]), np.array([[0, 0, 1]])]
points_y = [np.array([[0, 0, 0], [0, 0, 1]])]
```

We can generate distance matrices and marginals as follows.
```
eps = 1e-2
max_iter = 50
diff_thrsh = 1e-3
init = None

d_y, i_y = generate_d(points_y)
mu_y = [1/d_y.shape[0]] * d_y.shape[0]
mu_y = np.array(mu_y)

d_x, i_x = generate_d(points_x)
mu_x = [1/d_x.shape[0]] * d_x.shape[0]
mu_x = np.array(mu_x)
```

You can compute the transportation plan via the function `jgw_solver`, as follows.
```
mu, dif, costs = jgw_solver(mu_x, mu_y, d_x, d_y, i_x,
                            epsilon=epsilon, max_iter=max_iter,
                            diff_thrsh=diff_thrsh, init=init)
```

If you don't want to modify the marginals and distance matrices by hand we can use the weapper function instead as:
```
mu, dif, costs = jgw_wrapper(points_x, points_y, 
                            epsilon=epsilon, max_iter=max_iter,
                            diff_thrsh=diff_thrsh, init=init)
```

And check the convergence:
```
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 4))
plt.plot(dif)
plt.yscale('log')
plt.ylabel('error')
plt.xlabel('iteration')
plt.show()
```




## Data Generation
For generating the benchmarking dataset of models that we used in our experiments first go to `data/` and download the original atomic models  as follows.
```
python download.py
```

Then partition each map into its chains using the following script.
```
python splitter.py
```

Next we need to generate some hyperdata as follows.
```
python generate_hyper_data.py
```

Then convert atomic models into density maps using the `mrc_generator.py` script in ChimeraX. **You Can't perform this step outside of ChimeraX environment**.

After this step the dataset is ready to use.
