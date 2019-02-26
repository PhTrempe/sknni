# SkNNI
SkNNI (pronounced "skinny") stands for spherical k-nearest neighbors 
interpolation and is a geospatial interpolator.

## Setup
SkNNI may simply be installed from PyPI using `pip`.
```
pip install sknni
```

## Example
Here's a short usage example.
```
import numpy as np

from sknni import SkNNI

if __name__ == '__main__':
    observations = np.array([[30, 120, 20],
                             [30, -120, 10],
                             [-30, -120, 20],
                             [-30, 120, 0]])
    interpolator = SkNNI(observations)
    interp_coords = np.array([[30, 0],
                              [0, -120],
                              [0, 0],
                              [0, 120],
                              [-30, 0]])
    interpolation = interpolator(interp_coords)
    print(interpolation)
    # Output:
    # [[  30.          0.          9.312546]
    #  [   0.       -120.         14.684806]
    #  [   0.          0.         12.5     ]
    #  [   0.        120.         10.315192]
    #  [ -30.          0.         16.464548]]
```
