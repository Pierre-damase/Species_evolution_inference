"""
Define command-line options, arguments and sub-commands by using argparse.
"""

import argparse
import sys


def data_type(value):
    try:
        value = int(value)
    except ValueError as type_error:
        raise argparse.ArgumentTypeError('Value must be an integer !') from type_error

    if value < 1:
        raise argparse.ArgumentTypeError('Value must be an integer >= 1')

    return value


def arguments():
    """
    Define arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the subparser
    subparsers = parser.add_subparsers(dest='analyse', required=True)

    #############################################
    # Generate various set of data with msprime #
    #############################################
    data = subparsers.add_parser('data', help="Generate various unfolded sfs with msprime")
    data.add_argument(
        '--model', dest='model', required=True, choices=['decline', 'migration'],
        help="Kind of scenario to use for the generation of sfs with msprime - declin" \
        ", migration, etc."
    )

    group = data.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--value', dest='value', type=data_type,
        help="Simulation with msprime for a given tau & kappa"
    )

    sub_group = group.add_mutually_exclusive_group()
    sub_group.add_argument(
        '--snp', dest='snp', action='store_true',
        help="To determine and plot the snp distribution for a given model"
    )
    sub_group.add_argument(
        '--file', dest='file', action='store_true',
        help="Determine the length factor for each (tau, kappa) pairs"
    )

    #############################################
    # Msprime verification                      #
    #############################################
    msprime = subparsers.add_parser(
        'msprime', help="Check unfolded sfs generated with msprime for various scenarios"
    )

    #############################################
    # Optimisation                              #
    #############################################
    opt = subparsers.add_parser('opt', help="Compute optimisation of dadi's parameters")
    opt.add_argument(
        '--nb', dest='number', type=int, required=True,
        help="""Determine for a given number of sampled genomes n, the error rate of the
        inference of 100 observed for various mutation rate mu"""
    )

    #############################################
    # Inference of demographic history          #
    #############################################
    inf = subparsers.add_parser('inf', help="Compute inference of demographic history")
    tool = inf.add_mutually_exclusive_group(required=True)

    # Dadi
    tool.add_argument('-dadi', action='store_true',
                      help="Inference of demographic history with Dadi")
    inf.add_argument(
        '--model', dest="model", choices=['decline', 'migration'],
        required='--dadi' in sys.argv,  # required only if --dadi indicated
        help="Population model for the inference - model-constrained only, such as dadi"
    )
    inf.add_argument(
        '--opt', dest="opt", choices=['tau', 'kappa', 'tau-kappa'],
        required='--dadi' in sys.argv,  # required only if --dadi indicated
        help="Parameter to evaluate - (tau), (kappa), (kappa, tau)"
    )

    # Stairway
    tool.add_argument('-stairway', action='store_true',
                      help="Inference of demographic history with Stairway plot 2")

    #############################################
    # Plot error rate                           #
    #############################################
    er = subparsers.add_parser('er', help="Plot error rate of simulation with dadi")

    #############################################
    # Assessment of inferences                  #
    #############################################
    ases = subparsers.add_parser('ases', help="Evaluation of inference")
    # ases.add_argument(
    #     '--tool', dest='tool', required=True,
    #     help"Tools to evaluate - dadi, stairway plot, etc."
    # )
    ases.add_argument(
        '--param', dest='param', choices=['tau', 'kappa', 'tau-kappa'], required=True,
        help="Parameter to evaluate - (tau) - default, (kappa), (kappa, tau)"
    )

    # Plot SNPs distribution for various tau
    snp = subparsers.add_parser('snp', help="Plot SNPs distribution for various tau")

    # Inference with stairway plot
    stairway = subparsers.add_parser('stairway', help="Stairway plots")

    return parser.parse_args()
