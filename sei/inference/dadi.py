"""
This module allows the inference of demographic history of population with dadi.
"""

import dadi
import sys


def constant_model(ns, pts):
    """
    Constant model, i.e. population size is constant - control scenario.

    Parameter
    ---------
    ns: int
        the number of sampled genomes
    pts: list
        the number of grid points used in calculation
    """
    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi_ancestral, ns, (grid,))

    return sfs


def sudden_growth_model(ns, pts, *args):
    """
    Sudden growth model of the population.

    At time tau in the past, an equilibrium population of size nu undergoing a sudden growth,
    reaching size nu * kappa with kappa the force.

    Parameter
    ---------
    pop: int
        population size at time 0 (nowadays)
    """
    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid, nu=pop/kappa)

    # Define the sudden growth event at a time tau in past
    phi = dadi.Integration.one_pop(phi, grid, tau, nu=pop)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi, ns, (grid,))

    return sfs


def dadi_inference(pts_list, model_func, path="./Data/"):
    """
    Dadi inference.

    Parameter
    ---------
    pts_list: list
        the grid point use for extrapolation
    model_func: function
        the custom model_func

    Return
    ------
    ll_model: float
        likelihood of the data
    theta: float
        the optimal value of theta given the model
    """
    # Load the data
    sfs = dadi.Spectrum.from_file("{}/{}.fs".format(path, model_func.__name__))
    ns = sfs.sample_sizes

    # Ignore singletons - if yes
    ignore = 'no'
    if ignore == 'yes':
        sfs.mask[1] = True

    # Make the extrapolation version of our demographic model function
    model_func_extrapolated = dadi.Numerics.make_extrap_log_func(model_func)

    # Simulated frequency spectrum
    model = model_func_extrapolated(ns, pts_list)

    # Likelihood of the data
    ll_model = dadi.Inference.ll_multinom(model, sfs)

    # The optimal value of theta given the model
    theta = dadi.Inference.optimal_sfs_scaling(model, sfs)

    return ll_model, theta


if __name__ == "__main__":
   pass  # No actions desired
