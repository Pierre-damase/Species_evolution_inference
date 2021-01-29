"""
This module allows the inference of demographic history of population with dadi.
"""

import dadi
import sys


def constant_model(params, ns, pts):
    """
    Constant model, i.e. population size is constant - control scenario.

    Parameter
    ---------
    pts: list
        the number of grid points used in calculation
    """
    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Phi for the equilibrium ancestral population
    phi_ancestrale = dadi.PhiManip.phi_1D(grid)

    # Set up the model - constant population
    nu = 1  # population size constant over time

    phi = dadi.Integration.one_pop(phi_ancestrale, grid, params, nu=nu)

    # Finally, calculate the spectrum
    sfs = dadi.Spectrum.from_phi(phi, ns, (grid,))

    return sfs


def dadi_inference():
    # Load the data
    sfs = dadi.Spectrum.from_file("./Data/CST.fs")
    ns = sfs.sample_sizes

    # Ignore singletons - if yes
    ignore = 'no'
    if ignore == 'yes':
        sfs.mask[1] = True

    # These are the grid point settings will use for extrapolation
    pts_l = [300, 400, 500]

    # Our custom model function
    model_func = constant_model

    print(sfs.Watterson_theta())


def sudden_growth(params, ns, pts):
    T = params
    # Define the grid we'll use
    xx = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)

    # now do the population growth event
    Tb=1.0 # time of initial population size
    nu=0.01 # initial population size
    nuF=1.0 # final population size
    phi = dadi.Integration.one_pop(phi, xx, Tb, nu=nu)
    phi = dadi.Integration.one_pop(phi, xx, T, nu=nuF)

    # finally, calculate the spectrum
    sfs = dadi.Spectrum.from_phi(phi, ns, (xx,))

    return sfs


def main():
    Tb=1.0 # time of initial population size
    nu=1.0 # initial population size
    nuF=1.0 # final population size

    # Load the data
    data=dadi.Spectrum.from_file("./Data/CST.fs")
    ns=data.sample_sizes

    # ignore singletons
    ignore='no'
    if ignore=='yes':
	    data.mask[1]=True

    # These are the grid point settings will use for extrapolation
    pts_l=[300,400,500]

    # our custom model function
    func = sudden_growth

    # Now let's optimize parameters for this model

    # The upper_bound and lower_bound lists are for use in optimization.
    # Occasionally the optimizer will try wacky parameter values. We in particular
    # want to exclude values with very long times, very small population sizes, or
    # very high migration rates, as they will take a long time to evaluate.
    # Parameters are: (nu, T)
    upper_bound = [10]
    lower_bound = [0]

    # This is our initial guess for the parameters, which is somewhat arbitrary.
    p0 = [1]

    # Make the extrapolating version of our demographic model function.
    func_ex = dadi.Numerics.make_extrap_log_func(func)

    # Perturb our parameters before optimization. This does so by taking each
    # parameter a up to a factor of two up or down.
    p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
                                  lower_bound=lower_bound)

    # Do the optimization. By default we assume that theta is a free parameter,
    # since it's trivial to find given the other parameters. If you want to fix
    # theta, add a multinom=False to the call.
    # The maxiter argument restricts how long the optimizer will run. For real 
    # runs, you will want to set this value higher (at least 10), to encourage
    # better convergence. You will also want to run optimization several times
    # using multiple sets of intial parameters, to be confident you've actually
    # found the true maximum likelihood parameters.

    print('Beginning optimization ************************************************')
    popt = dadi.Inference.optimize_log(p0, data, func_ex, pts_l, 
                                       lower_bound=lower_bound,
                                       upper_bound=upper_bound,
                                       verbose=1, maxiter=3)

    # The verbose argument controls how often progress of the optimizer should be
    # printed. It's useful to keep track of optimization process.
    print('Finished optimization **************************************************')

    print('Best-fit parameters: {0}'.format(popt))

    # Calculate the best-fit model AFS.
    model = func_ex(popt, ns, pts_l)

    # Likelihood of the data given the model AFS.
    ll_model = dadi.Inference.ll_multinom(model, data)
    print('Maximum log composite likelihood: {0}'.format(ll_model))

    # The optimal value of theta given the model.
    theta = dadi.Inference.optimal_sfs_scaling(model, data)
    print('Optimal value of theta: {0}'.format(theta))


if __name__ == "__main__":
    sys.exit()  # No actions desired
