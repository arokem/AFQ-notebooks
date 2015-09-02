import numpy as np
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

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

def affine_registration(moving, static,
                        moving_grid2world=None,
                        static_grid2world=None,
                        nbins = 32,
                        sampling_prop = None,
                        metric=MutualInformationMetric,
                        level_iters = [10000, 1000, 100],
                        sigmas = [3.0, 1.0, 0.0],
                        factors = [4, 2, 1]):
    """
    Create an affine registration between a moving and a static image.
    
    """
    # Initialize our registration class instance with the metric:
    affreg = AffineRegistration(metric=metric(nbins, sampling_prop),
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)
    # Start by aligning centers of mass:
    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)

    # Use that as a starting point to calculate a translation:
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    # Which is then used as a starting point for a rigid:
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    # Finally used as  staring point for the affine:
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)
