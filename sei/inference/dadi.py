"""
This module allows the inference of demographic history of population with dadi.
"""

import dadi
import sys


OPTIMIZATION = None


def constant_model(ns, pts):
    """
    Constant model, i.e. population size is constant - control scenario.

    It's the standard neutral model.

    Parameter
    ---------
    ns: int
        the number of sampled genomes in resulting spectrum
    pts: list
        the number of grid points to use in integration
    """
    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi_ancestral, ns, (grid,))

    return sfs


def params_decline_model(params):
    global OPTIMIZATION
    if OPTIMIZATION == "tau":
        return 10, params  # Fixe kappa
    elif OPTIMIZATION == "kappa":
        return params, 1.0  # Fixe tau
    else:
        return params


def sudden_decline_model(params, ns, pts):
    """
    Sudden decline model of the population.

    At time tau in the past, an equilibrium population of size nu undergoing a sudden decline,
    reaching size nu * kappa with kappa the force.

    Parameter
    ---------
    params
      - kappa: ratio of contemporary population's size to ancient population's size
      - tau: time in the past at which size change happened
    ns: int
        the number of sampled genomes in resulting spectrum
    pts: list
        the number of grid points to use in integration
    """
    # Params (kappa, tau)
    kappa, tau = params_decline_model(params)

    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid, nu=1.0*kappa)

    # Define the sudden decline event at a time tau in past
    phi = dadi.Integration.one_pop(phi_ancestral, grid, tau, nu=1.0)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi, ns, (grid,))

    return sfs


def params_migration_model(params):
    global OPTIMIZATION
    if OPTIMIZATION == "migration":
        return params, 0  # No migration from population 1 to population 2


def two_pops_migration_model(params, ns, pts):
    """

    Parameter
    ---------
    params
      - kappa: ratio of population's 1 size to population's 2 size
      - tau: time in the past of split
      - m12: migration rate from population 2 to 1
      - m21: migration rate from population 1 to 2
    ns: int
        the number of sampled genomes in resulting spectrum
    pts: list
        the number of grid points to use in integration
    """
    kappa, tau, m12, m21 = params_migration_model(params)

    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid)

    # Define the sudden decline event at a time tau in past
    phi = dadi.Integration.two_pops(phi_ancestral, grid, tau, nu1=1.0, nu2=1.0*kappa, m12=m12,
                                    m21=m21)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi, ns, (grid, grid))

    return sfs


def parameters_optimization(p0, sfs, model_func, pts_list, lower_bound, upper_bound, verbose=0):
    """
    Parameters optimization.

    The upper_bound and lower_bound lists are use in optimization. Occasionally the optimizer
    will try wacky parameters values. We in particular want to exclude values with very long
    times, very small population sizes, or very high migration rates, as they will take a long
    time to evaluate.
    Parameters can be (kappa), (tau), (kappa, tau), etc.

    Parameter
    ---------
    p0: list
        Initial parameters - this is our initial guess, which is somewhat arbitrary.
    sfs
        Spectrum with data
    model_func
        Function to evaluate model spectrum - extrapolated.
    pts_list: list
        the grid point use for extrapolation
    lower_bound: list
        Lower bound on parameter values. If not None, must be of same length as p0
    upper_bound: list
        Upper bound on parameter values. If not None, must be of same length as p0

    Return
    ------
    popt: list
        Optimize log(params) to fit model to data using the BFGS method.
    """
    # Perturb our parameters before optimization. This does so by taking each parameter a up
    # to a factor of two up or down.
    p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)

    # Do the optimization. By default we assume that theta is a free parameter, since it's
    # trivial to find given the other parameters. If you want to fix theta, add a
    # multinom=False to the call.
    # The maxiter argument restricts how long the optimizer will run. For real runs, you will
    # want to set this value higher (at least 10), to encourage better convergence. You will
    # also want to run optimization several times using multiple sets of intial parameters, to
    # be confident you've actually found the true maximum likelihood parameters.

    if verbose:
        print('Beginning optimization ************************************************')

    popt = dadi.Inference.optimize_log(p0, sfs, model_func, pts_list,
                                       lower_bound=lower_bound, upper_bound=upper_bound,
                                       verbose=verbose, maxiter=3)

    if verbose:
        # The verbose argument controls how often progress of the optimizer should be
        # printed. It's useful to keep track of optimization process.
        print('Finished optimization **************************************************')
        print('Best-fit parameters: {}'.format(popt))

    return popt


def dadi_inference(pts_list, model_func, opt=None, verbose=0, path="./Data/", name="SFS"):
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
    model: list
        the sfs inferred
    """
    global OPTIMIZATION
    OPTIMIZATION = opt

    # Load the data
    sfs = dadi.Spectrum.from_file("{}{}.fs".format(path, name))
    ns = sfs.sample_sizes

    # Make the extrapolation version of our demographic model function
    model_func_extrapolated = dadi.Numerics.make_extrap_log_func(model_func)

    # Optimisation of model parameters
    if OPTIMIZATION == "tau":
        # Param: (tau)
        p0, lower_bound, upper_bound = [0.9], [1e-4], [1e4]
        popt = parameters_optimization(p0, sfs, model_func_extrapolated, pts_list, lower_bound,
                                       upper_bound, verbose=verbose)
        # Simulated frequency spectrum
        model = model_func_extrapolated(popt, ns, pts_list)

    elif OPTIMIZATION == "kappa":
        # Param: (kappa)
        p0, lower_bound, upper_bound = [1.0], [1e-4], [1.5e3]
        popt = parameters_optimization(p0, sfs, model_func_extrapolated, pts_list, lower_bound,
                                       upper_bound, verbose=verbose)
        # Simulated frequency spectrum
        model = model_func_extrapolated(popt, ns, pts_list)

    elif OPTIMIZATION == "tau-kappa":
        # Params: (kappa, tau)
        p0, lower_bound, upper_bound = [1.0, 0.9], [1e-4, 1e-4], [1.5e3, 1e4]
        popt = parameters_optimization(p0, sfs, model_func_extrapolated, pts_list, lower_bound,
                                       upper_bound, verbose=verbose)
        # Simulated frequency spectrum
        model = model_func_extrapolated(popt, ns, pts_list)

    elif OPTIMIZATION == "migration":
        p0, lower_bound, upper_bound = [10.0, 1.0, 1.0], [1.0, 1e-4, 0.0], [1e6, 1e5, 10.0]
        popt = parameters_optimization(p0, sfs, model_func_extrapolated, pts_list, lower_bound,
                                       upper_bound, verbose=verbose)
        # Simulated frequency spectrum
        model = model_func_extrapolated(popt, ns, pts_list)

    else:
        # Simulated frequency spectrum
        model = model_func_extrapolated(ns, pts_list)

    # Log-likelihood of the data (sfs) given the model
    ll_model = dadi.Inference.ll_multinom(model, sfs)

    # The optimal value of theta given the model
    theta = dadi.Inference.optimal_sfs_scaling(model, sfs)

    return ll_model, theta, model


if __name__ == "__main__":
    sys.exit()  # No actions desired
