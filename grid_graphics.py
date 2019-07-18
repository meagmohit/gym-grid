import numpy as np


arr_cell =  np.array([
    [ 1., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  1.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  1.],
    [ 1., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  1.,  1.],
])

arr_target =  np.array([
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  0.,  0.],
    [ 0., 0.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  1., 1., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
])


arr_arrow =  np.array([
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 1., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 1., 1., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 1., 1., 1.,  0.,  0.,  0.,  0.],
    [ 1., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  0.,  0.,  0.],
    [ 1., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  0.,  0.],
    [ 1., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 1., 1., 1.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 1., 1., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 1., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.]
])

arr_ghost =  np.array([
    # [ 3., 3.,  3.,  3.,  3.,  3.,  3.,  3., 3., 3., 3., 3.,  3.,  3.,  3.,  3.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    [ 0., 3.,  3.,  3.,  3.,  3.,  0.,  0., 0., 3., 3., 3.,  3.,  3.,  3.,  0.],
    [ 0., 3.,  3.,  3.,  0.,  0.,  1.,  1., 1., 0., 0., 3.,  3.,  3.,  3.,  0.],
    [ 0., 3.,  3.,  0.,  1.,  1.,  1.,  1., 1., 1., 1., 0.,  3.,  3.,  3.,  0.],
    [ 0., 3.,  0.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  0.,  3.,  3.,  0.],
    [ 0., 0.,  1.,  1.,  1.,  1.,  3.,  3., 1., 1., 1., 1.,  3.,  3.,  3.,  0.],
    [ 0., 0.,  1.,  1.,  1.,  3.,  2.,  2., 1., 1., 1., 3.,  2.,  2.,  3.,  0.],
    [ 0., 0.,  1.,  1.,  1.,  3.,  2.,  2., 1., 1., 1., 3.,  2.,  2.,  3.,  0.],
    [ 0., 1.,  1.,  1.,  1.,  1.,  3.,  3., 1., 1., 1., 1.,  3.,  3.,  0.,  0.],
    [ 0., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  0.,  0.],
    [ 0., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  0.,  0.],
    [ 0., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  0.,  0.],
    [ 0., 1.,  1.,  1.,  1.,  1.,  1.,  1., 1., 1., 1., 1.,  1.,  1.,  0.,  0.],
    [ 0., 1.,  0.,  1.,  1.,  1.,  0.,  1., 1., 1., 1., 0.,  1.,  1.,  0.,  0.],
    [ 0., 0.,  3.,  0.,  1.,  0.,  3.,  0., 1., 1., 0., 3.,  0.,  1.,  0.,  0.],
    [ 0., 0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0., 0., 0.,  0.,  0.,  0.,  0.],
    # [ 0., 3.,  3.,  3.,  0.,  3.,  3.,  3., 0., 0., 3., 3.,  3.,  0.,  0.,  3.],
])
