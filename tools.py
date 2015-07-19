import numpy as np
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration

metric_dict = {'CC': CCMetric,
               'EM': EMMetric,
               'SSD': SSDMetric}

def syn_registration(moving, static, moving_grid2world=None, static_grid2world=None,
                     metric='CC', dim=3, level_iters = [10, 10, 5], prealign=None):
    """
    Register a source image (moving) to a target image (static)
    Parameters
    ----------
    moving : ndarray
        The source image data to be registered
    moving_grid2world : array, shape (4,4)
        The affine matrix associated with the moving (source) data.
    static : ndarray
        The target image data for registration
    static_grid2world : array, shape (4,4)
        The affine matrix associated with the static (target) data
    metric : string, optional
        The metric to be optimized. One of `CC`, `EM`, `SSD`, Default: CCMetric.
    dim: int (either 2 or 3), optional
       The dimensions of the image domain. Default: 3
    level_iters : list of int, optional
        the number of iterations at each level of the Gaussian Pyramid (the
        length of the list defines the number of pyramid levels to be
        used).
    Returns
    -------
    warped_moving : ndarray
        The data in `moving`, warped towards the `static` data.
    forward : ndarray (..., 3)
        The vector field describing the forward warping from the source to the target.
    backward : ndarray (..., 3)
        The vector field describing the backward warping from the target to the source
    """
    use_metric = metric_dict[metric](dim)

    sdr = SymmetricDiffeomorphicRegistration(use_metric, level_iters)
    mapping = sdr.optimize(static, moving, static_grid2world=static_grid2world,
                            moving_grid2world=moving_grid2world, prealign=prealign)

    warped_moving = mapping.transform(moving)
    return warped_moving, mapping
