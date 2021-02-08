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


def sudden_decline_model(params, ns, pts):
    """
    Sudden growth model of the population.

    At time tau in the past, an equilibrium population of size nu undergoing a sudden growth,
    reaching size nu * kappa with kappa the force.

    Parameter
    ---------
    pop: int
        population size at time 0 (nowadays)
    """
    tau, kappa = params, 10

    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid, nu=1.0*kappa)

    # Define the sudden decline event at a time tau in past
    phi = dadi.Integration.one_pop(phi_ancestral, grid, tau, nu=1.0)

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
    sfs = dadi.Spectrum.from_file("{}{}.fs".format(path, model_func.__name__))
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


def dadi_inference_tau(pts_list, model_func, path="./Data/"):
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
    sfs = dadi.Spectrum.from_file("{}{}.fs".format(path, model_func.__name__))
    ns = sfs.sample_sizes

    # Ignore singletons - if yes
    ignore = 'no'
    if ignore == 'yes':
        sfs.mask[1] = True

    # Now let's optimize parameters for this model

    # The upper_bound and lower_bound lists are for use in optimization.
    # Occasionally the optimizer will try wacky parameter values. We in particular want to
    # exclude values with very long times, very small population sizes, or very high migration
    # rates, as they will take a long time to evaluate.
    # Parameters are: (tau)
    upper_bound = [10]
    lower_bound = [0]

    # This is our initial guess for the parameters, which is somewhat arbitrary.
    p0 = [1]

    # Make the extrapolation version of our demographic model function
    model_func_extrapolated = dadi.Numerics.make_extrap_log_func(model_func)

    # Perturb our parameters before optimization. This does so by taking each
    # parameter a up to a factor of two up or down.
    p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)

    # Do the optimization. By default we assume that theta is a free parameter, since it's
    # trivial to find given the other parameters. If you want to fix theta, add a
    # multinom=False to the call.
    # The maxiter argument restricts how long the optimizer will run. For real runs, you will
    # want to set this value higher (at least 10), to encourage better convergence. You will
    # also want to run optimization several times using multiple sets of intial parameters, to
    # be confident you've actually found the true maximum likelihood parameters.

    print('Beginning optimization ************************************************')
    tau = dadi.Inference.optimize_log(p0, sfs, model_func_extrapolated, pts_list,
                                      lower_bound=lower_bound, upper_bound=upper_bound,
                                      verbose=1, maxiter=3)
                                   
    # The verbose argument controls how often progress of the optimizer should be
    # printed. It's useful to keep track of optimization process.
    print('Finished optimization **************************************************')

    print('Best-fit parameters: {}'.format(tau))

    # Simulated frequency spectrum
    model = model_func_extrapolated(tau, ns, pts_list, )

    # Likelihood of the data
    ll_model = dadi.Inference.ll_multinom(model, sfs)

    # The optimal value of theta given the model
    theta = dadi.Inference.optimal_sfs_scaling(model, sfs)

    return ll_model, theta


if __name__ == "__main__":
   pass  # No actions desired
