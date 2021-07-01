"""
This module allows you to create graphics.
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D


def normalization(sfs):
    """
    Data normalization to (0,1).
    """
    somme = sum(sfs)
    return [ele / somme for ele in sfs]


######################################################################
# Heatmap                                                            #
######################################################################

def heatmap_axis(ax, xaxis, yaxis, cbar, lrt=False):
    """
    Heatmap customization.

    Parameter
    ---------
    ax: matplotlib.axes.Axes
        ax to modify
    xaxis: str
        x-axis label
    yaxis: str
        y-axis label
    cbar: str
        colormap label
    """
    # Name
    names = [
        "$Log_{10}$" + "({})".format('τ' if xaxis == 'Tau' else xaxis),
        "$Log_{10}$" + "({})".format('κ' if yaxis == 'Kappa' else yaxis)
    ]  # (xaxis, yaxis)

    # x-axis
    plt.xticks(
        np.arange(64, step=7) + 0.5,
        labels=[round(ele, 2) for ele in np.arange(-4, 2.5, 0.7)],
        rotation='horizontal'
    )
    plt.xlabel(names[0], fontsize="large")

    # y-axis
    # ax.set_ylim(ax.get_ylim()[::-1])  # reverse y-axis
    plt.yticks(
        np.arange(64, step=7) + 0.5,
        labels=[round(ele, 2) for ele in np.arange(-3.5, 3, 0.7)]
    )
    plt.ylabel(names[1], fontsize="large")

    # Set colorbar label & font size
    ax.figure.axes[-1].set_ylabel(cbar, fontsize="large")
    if lrt:
        ax.collections[0].colorbar.set_ticks([0, 20., 40., 60., 80., 100.])
        ax.collections[0].colorbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    # Set axis label of kappa = 0.0, i.e. constant model
    index = [
        i for i, ele in enumerate(plt.gca().get_yticklabels()) if ele.get_text() == "0.0"
    ][0]  # get the index of kappa = 0.0
    plt.gca().get_yticklabels()[index].set_color('#8b1538')  # set the color
    plt.gca().get_yticklabels()[index].set_fontsize('medium')  # set the size
    plt.gca().get_yticklabels()[index].set_fontweight('bold')  # set the weight

    # Hlines for kappa = 0
    ax.hlines([35, 36], *ax.get_xlim(), colors="white", lw=2.)
    ax.vlines([0, 65], ymin=35, ymax=36, color="white", lw=2.)


def plot_heatmap(data, title, cbar, filout="./Figures/heatmap.png", lrt=False):
    """
    Heatmap

    Parameter
    ---------
    data: pandas DataFrame
    """
    # Set up plot
    plt.figure(figsize=(12, 9), constrained_layout=True)
    sns.set_theme(style='whitegrid')

    # Pre-processing data
    df = data.pivot(index=data.columns[1], columns=data.columns[0], values=data.columns[2])

    # Plot
    ax = sns.heatmap(df, cmap='viridis')

    # Heatmap x and y-axis personnalization
    heatmap_axis(ax=ax, xaxis=df.columns.name, yaxis=df.index.name, cbar=cbar, lrt=lrt)

    # Title
    plt.title(title, fontsize="large", fontweight='bold', pad=10.5)

    plt.savefig(filout, format='png', dpi=150)
    plt.plot()


######################################################################
# SFS shape verification                                             #
######################################################################

def sfs_label(length, title, save=False):
    """
    Set up sfs caption, label and title.

    Parameter
    ---------
    length: length of the SFS
    title: title of the plot
    """
    # Caption
    plt.legend(loc="upper right", fontsize="x-large")

    # Label axis
    if save:
        plt.xlabel("Fréquence allélique", fontsize="xx-large")
        plt.ylabel("SNPs - pourcentage", fontsize="xx-large")
    else:
        plt.xlabel("Allele frequency", fontsize="x-large")
        plt.ylabel("Percent of SNPs", fontsize="x-large")

    # X axis values
    xtick_pas = 1 if length <= 10 else 2 if length < 20 else length % 10 + 1

    x_ax, x_values = [], []
    for i in range(0, length, xtick_pas):
        x_ax.append(i)
        x_values.append("{}/{}".format(i+1, length+1))
    plt.xticks(x_ax, x_values, fontsize="x-large")

    # Title + show
    plt.title(title, fontsize="xx-large", fontweight='bold', y=1.01)


# For observed SFS generated with msprime

def plot_sfs(data, save=False):
    """
    Graphic representation of Site Frequency Spectrum (SFS), save to the folder ./Figures.

    Parameter
    ---------
    data: tupple
        - 0: dictionary of {model: sfs}
        - 1: dictionary of {model: model_parameter}, with  model_parameter either (tau, kappa)
          or (m12, kappa)
        - 2: dictionary of msprime simulation parameters

    save: boolean
        If set to True save the plot in ./Figures/SFS-shape
    """
    color = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:gray"]
    labels = ['Modèle constant', 'Modèle théorique constant', 'Modèle déclin',
              'Modèle croissance']

    # Set up plot
    plt.figure(figsize=(12, 9))  #, constrained_layout=True)

    cpt = 0
    for key, sfs in data[0].items():
        # Normalization of SFS - sum to 1
        normalized_sfs = [ele / sum(sfs) for ele in sfs]

        # Label
        label = labels[cpt] if save else key
        if key == 'Theoretical model':
            label += " - Fu, 1995"
        elif not key == 'Constant model':
            label += " - "
            for param, value in data[1][key].items():
                label += "{}={}".format(
                    'τ' if param == 'Tau' else 'κ' if param == 'Kappa' else param, value
                )
                if param != list(data[1][key].keys())[-1]:
                    label += ", "

        # Plot
        with plt.style.context('seaborn-whitegrid'):  # use seaborn style for plot
            plt.plot(normalized_sfs, color=color[cpt], label=label,
                     marker='o' if key == 'Theoretical model' else '')

        cpt += 1

    # Label, caption and title
    if save:
        title = (
            "SFS non plié pour différents scénarios avec Ne={}, mu={}, taux de recombinaison={}"
            " & L={:.1e}"
        ).format(data[2]['Ne'], data[2]['mu'], data[2]['rcb_rate'], data[2]['length'])
    else:
        title = (
            "Unfold SFS with Ne={}, mutation rate mu={}, recombination rate={} & L={:.1e}"
        ).format(data[2]['Ne'], data[2]['mu'], data[2]['rcb_rate'], data[2]['length'])

    sfs_label(length=len(sfs), title=title, save=save)

    if save:
        plt.savefig('./Figures/SFS/SFS-shape.png', format='png', dpi=150)
    plt.show()
    plt.clf()


# For estimated SFS generated with Dadi

def compute_theoretical_sfs(length):
    """
    compute the theoretical sfs of any constant population.
    """
    theoretical_sfs = [0] * (length)
    for i in range(length):
        theoretical_sfs[i] = 1 / (i+1)
    return theoretical_sfs


def plot_sfs_inference(data, parameters, colors, suptitle):
    """
    Plot SFS from inferences with Dadi - file are in ./Data/Dadi/

    Parameter
    ---------
    data: pandas DataFrame of dadi inferences
    parameters: list of pairs (tau, kappa) or (m12, kappa)
    """
    # Set up title
    title = ["Observed SFS generated with  msprime", "Estimated SFS with Dadi"]

    # Set up plot
    _, axs = plt.subplots(1, 2, figsize=(20, 8))  # (width, height)

    cpt = 0
    for _, row in data.iterrows():

        param = {
            k: round(np.log10(v), 2) for k, v in row['Parameters'].items() if k != 'Theta'
        }
        if param in parameters:
            # Label
            label = "SFS - {}" \
                .format({k: "{:.1e}".format(v) for k, v in row['Parameters'].items()
                         if k != 'Theta'})

            # Plot
            with plt.style.context('seaborn-whitegrid'):  # use seaborn style for plot
                axs[0].plot(normalization(row['SFS observed'][0]), label=label,
                            color=colors[cpt])
                axs[1].plot(normalization(row['M1']['SFS'][0]), label=label,
                            color=colors[cpt])

            cpt += 1

    # Compute theoretical SFS of any constant population
    theoretical_sfs = compute_theoretical_sfs(len(row['SFS observed'][0]))

    # Compute X axis ticks & values
    x_ax, x_values = [], []
    for i in range(10):
        x_ax.append(i+i)
        x_values.append("{}/{}".format(i*2+1, len(theoretical_sfs)+1))

    # Plot
    for i, ax in enumerate(axs):
        # Plot theoretical SFS of any constant population - control SFS
        with plt.style.context('seaborn-whitegrid'):  # use seaborn style fot plot
            ax.plot(normalization(theoretical_sfs), color="tab:orange", marker="o",
                    label="Theoretical SFS")

        # Label axis
        ax.set_xlabel("Allele frequency", fontsize="large")
        ax.set_ylabel("Percent of SNPs", fontsize="large")

        # X axis ticks & labels
        ax.set_xticks(x_ax)
        ax.set_xticklabels(x_values)

        # Title
        ax.set_title(title[i], fontsize="large")

        # Caption
        ax.legend(loc="upper right", fontsize="medium")

    # Suptitle
    plt.suptitle(suptitle, fontsize="xx-large")

    plt.show()
    plt.clf()


# For SFS of real data

def plot_species_sfs(species, values, save=False):
    # Set up plot
    plt.figure(figsize=(12, 9), constrained_layout=True)

    # Color
    color = {"Increasing": 'tab:green', "Decreasing": 'tab:red', "Stable": 'tab:blue',
             "ras": 'tab:gray'}

    # Plot SFS
    espece = "{} {}".format(species.split(' ')[0], species.split(' ')[1].lower())
    with plt.style.context('seaborn-whitegrid'): 
        plt.plot(normalization(values['SFS']), label=espece, color=color[values['Status']])

    # Plot theoritical SFS of any constant population
    theoretical_sfs = compute_theoretical_sfs(len(values['SFS']))
    with plt.style.context('seaborn-whitegrid'): 
        plt.plot(normalization(theoretical_sfs), label="Modèle théorique constant - Fu, 1995",
                 color="tab:orange", linestyle="dashed")

    # Label, caption and title
    sfs_label(length=len(theoretical_sfs), title="SFS de l'espèce {}".format(espece), save=save)

    if save:
        plt.savefig("./Figures/SFS/{}.png".format("_".join(species.split(' '))), format='png',
                    dpi=150)
    plt.show()
    plt.clf()


######################################################################
# Plot for the optimization of grid size                             #
######################################################################

def plot_optimisation_grid(data, log_scale):
    """
    Plot for a given scenario the likelihood and optimal theta's value for various grid size.

    Parameter
    ---------
    data: dictionary
      - mu the rate of mutation
        - Likelihood
        - Estimated theta 
        - Theoretical theta

    log_scale:
    """
    fig, axs = plt.subplots(2, 2, sharex=True)

    cpt = 0
    mu, color = list(data.keys()), ['tab:red', 'tab:blue', 'tab:orange']

    for i in range(2):
        for j in range(2):
            ##############################################
            # Plots with different scale on the same fig #
            ##############################################

            # Set x_axis value
            x_ax = []
            for k in range(len(log_scale)):
                x_ax.append(k)
            axs[i, j].set_xticks(x_ax)  # value
            axs[i, j].set_xticklabels(log_scale)  # labels

            # Likelihood plot
            axs[i, j].plot(data[mu[cpt]]["Likelihood"], color=color[0])
            axs[i, j].tick_params(axis='y', labelcolor=color[0])

            # Theta plot
            ax2 = axs[i, j].twinx()  # instantiate a second axes that shares the same x-axis
            ax2.plot(data[mu[cpt]]["Estimated theta"], color=color[1])
            ax2.plot(data[mu[cpt]]["Theoretical theta"], color=color[2], linestyle='dashed')
            ax2.tick_params(axis='y', labelcolor=color[1])

            # Optimal value for grid size
            axs[i, j].axvline(x=4, color="tab:gray", linestyle="dashed")

            # Suptitle of each plot
            axs[i,j].set_title("Mutation rate {}".format(mu[cpt]))

            cpt +=1

    # Add common legend
    lg_ele = [
        Line2D([0], [0], linestyle='', marker='.', color=color[0], label='Likelihood'),
        Line2D([0], [0], linestyle='', marker='.', color=color[1], label='Estimated theta'),
        Line2D([0], [0], linestyle='', marker='.', color=color[2], label='Theoretical theta')
    ]
    legend = fig.legend(handles=lg_ele, loc='upper center', ncol=3, fontsize='medium',
                        bbox_to_anchor=(0., 1.05, 1., .102), borderaxespad=0.)
    fig.gca().add_artist(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Add xlabel for bottom plot only
    cpt = 0
    for ax in axs.flat:
        if cpt > 1:
            ax.set(xlabel='Grid scale')
        cpt += 1

    # Title + save plot to the folder ./Figures
    fig.suptitle("Likelihood & theta's value for various grid point size", fontsize="x-large",
                 y=-0.05)
    plt.savefig("./Figures/optimisation_grid", bbox_inches="tight", dpi=300)
    plt.clf()


######################################################################
# Plot of error rate of dadi                                         #
######################################################################

def plot_error_rate(sample):
    """
    Plot the error rate of theta estimated for 100 inference with dadi.
    """
    # Read a csv file into a pandas DataFrame
    data = pd.read_csv("./Data/Error_rate/error-rate-{}.csv".format(sample), sep='\t')

    # Round value in execution time - some values not round for an unexpected reason
    tmp = pd.DataFrame(
        {"Execution time": [round(ele, 3) for ele in data['Execution time'].to_list()]}
    )
    data.update(tmp)

    # Plot
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x="mu", y="Error rate", hue="Execution time", data=data,
                     width=0.45, dodge=False)

    # Set yaxis range
    ax.set(ylim=(0.85, 1.15))

    # Legend out of the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize='small',
              borderaxespad=0., title="Average run time")

    # Title + save plot to folder ./Figures
    title = \
        """Error rate of 100 inferences with dadi for various mutation rate mu
        with n={} genomes sampled\n
        Each simulation is a constant model population
        """.format(sample)

    plt.title(title, fontsize="large", loc="center", wrap=True)
    plt.savefig("./Figures/Error_rate/error-rate-{}".format(sample), bbox_inches="tight")
    plt.clf()


######################################################################
# SNPs distribution                                                  #
######################################################################

def data_preprocessing(observed):
    """
    Parameter
    ---------
    observed: pandas DataFrame
        observed data (SFS, parameters) for a given scenario

    Return
    ------
    data: pandas DataFrame
        reshaped DataFrame organized by given index (Kappa) / column (Tau) values
    """
    # New pandas DataFrame
    data = pd.DataFrame()

    # Compute log10 of parameters - either (tau, kappa) or (m12, kappa)
    keys = observed['Parameters'][0].keys()
    names = []
    for key in keys:
        if key in ['Tau', 'Kappa', 'm12']:
            names.append(key)
            data[key] = observed['Parameters'].apply(lambda param: param[key])

    # Compute mean(SNPs)
    data['SNPs'] = observed['SNPs'].apply(lambda snp: np.log10(np.mean(snp)))

    return data.pivot(names[1], names[0], 'SNPs')


def plot_snp_distribution(model, filin, path_data):
    """
    Parameter
    ---------
    data:
        data to plot
    model:
        either decline or migration
    """
    observed = pd.read_json("{}{}".format(path_data, filin), typ='frame')
    data = data_preprocessing(observed)

    # Set-up plot
    plt.figure(figsize=(12,9), constrained_layout=True)
    sns.set_theme(style='whitegrid')

    # Plot
    ax = sns.heatmap(data, cmap="viridis")

    # Heatmap x and y-axis personnalization
    heatmap_axis(ax=ax, xaxis=data.columns.name, yaxis=data.index.name,
                 cbar="SNPs - log scale")

    # Title
    title = "SNPs distribution in terms of {} & {} - {} model" \
        .format(data.columns.name, data.index.name, model)
    plt.title(title, fontsize="x-large", color="#8b1538")

    plt.show()


######################################################################
# Dadi inference                                                     #
######################################################################

# Weighted square distance #

def plot_weighted_square_distance(data, fixed, labels, suptitle):
    """
    Lineplot of weighted square distance for a fixed parameter.

    Parameter
    ---------
    data: pandas DataFrame of inference with Dadi
    fixed: fixed parameter - eitehr tau, kappa or m12
    labels: list of label for the plot
    suptitle: suptitle of the plot
    """
    d2 = ['d2 observed inferred', 'd2 models']
    title = [
        'd2 between the observed SFS and inferred one with M1',
        'd2 between the inferred SFS of two models (M0 & M1)'
    ]

    # Set-up plot
    sns.set_theme(style='whitegrid')

    # Sub-plots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Figsize: (width, height)

    # Plot
    for i, ax in enumerate(axs):  # Iterate through subplot

        for j, dataframe in enumerate(data):  # Iterate through DataFrame in data
            # Data
            df = pd.DataFrame()
            df[fixed] = \
                dataframe['Parameters'].apply(lambda param: round(np.log10(param[fixed]), 2))
            df[d2[i]] = dataframe[d2[i]].apply(np.log10)

            # Plot
            _ = sns.lineplot(x=fixed, y=d2[i], data=df, label=labels[j], ax=ax)

        ax.legend(fontsize="large")
        ax.set_title(title[i], fontsize="large")

    plt.suptitle(suptitle, fontsize="x-large")

    plt.plot()


# Log-likelihood ratio test #

def plot_likelihood(data, fixed, labels, suptitle):
    """
    Lineplot of log-likelihood ratio test for a fixed parameter.

    Parameter
    ---------
    data: pandas DataFrame of inference with Dadi
    fixed: fixed parameter - eitehr tau, kappa or m12
    labels: list of label for the plot
    suptitle: suptitle of the plot
    """
    # Set-up plot
    plt.figure(figsize=(10,8), constrained_layout=True)
    sns.set_theme(style='whitegrid')

    # Plot
    for i, dataframe in enumerate(data):
        # Data
        df = pd.DataFrame()
        df[fixed] = dataframe['Parameters'].apply(lambda param: np.log10(param[fixed]))
        df['Positive hit'] = dataframe['Positive hit'].apply(np.float64)

        # Plot
        ax = sns.lineplot(x=fixed, y='Positive hit', data=df, label=labels[i])

    ax.legend(fontsize="large")

    plt.suptitle(suptitle, fontsize="x-large")

    plt.plot()


# Evaluation of estimated parameters #

def plot_parameters_evaluation(data, key, fixed):
    """
    Plot le paramètre estimée en fonction du paramètre estimée.

    Parameters
    ----------
    data: pandas DataFrame of inference with Dadi
    key: the parameters to check - either Tau, Kappa, m12 or Theta
    fixed: pair of (param, value) - for the title of the plot
    """
    # Extract the parameter key (observed and estimated) from data
    parameters = extract_parameters(data, key)

    # Set up plot
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 7))

    # Plot
    with plt.style.context('seaborn-whitegrid'):  # use seaborn style for plot
        plt.plot(np.log10(parameters['Observed']), np.log10(parameters['Estimated']),
                 marker='.', linestyle="")
        plt.plot(np.log10(parameters['Observed']), np.log10(parameters['Observed']))

    # Label
    plt.xlabel("{} observés (log10)".format(key), fontsize="x-large")
    plt.ylabel("{} estimés (log10)".format(key), fontsize="x-large")

    # Title
    title = "{} observés en fonction de {} estimés pour log({}) = {}" \
        .format(key, key[0].lower() + key[1:], fixed[0], fixed[1])
    plt.title(title, fontsize="xx-large")

    # Plot vertical line for log(kappa) = 0
    if key == 'Kappa':
        plt.axvline(0, color="#8b1538")

    plt.plot()


# Plot some inferred (dadi) or observed (msprime) SFS

def get_data(df, sfs):
    """
    Return
    ------
    data: dico
      - key: for which data we want to plot the SFS
      - value: list of {Parameters used with msprime, SFS observed}
    """
    parameters = [0., 1., -1.]
    data = {key: [] for key in parameters}
    
    for _, row in df.iterrows():
        param = {
            k: round(np.log10(v), 2) for k, v in row['Parameters'].items() if k != 'Theta'
        }
        if param['Tau'] in parameters and param['Kappa'] in parameters:
            if sfs == 'observed':
                data[param['Tau']].append({'Parameters': param, 'SFS': row['SFS observed'][0]})
            else:
                data[param['Tau']].append({'Parameters': param, 'SFS': row['M1']['SFS'][0]})

    return data


def plot_sfs(data, ax, title, xaxis=True):
    for value in data:
        # Normalization of the SFS
        normalized_sfs = normalization(value['SFS'])
            
        # Label and color
        if value['Parameters']['Kappa'] > 0:
            label = 'Scénario de décroissance avec $log_{10}(κ) = $' \
                + '{}'.format(value['Parameters']['Kappa'])
            color = 'red'
        elif value['Parameters']['Kappa'] < 0:
            label = 'Scénario de croissance avec $log_{10}(κ) = $' \
                + '{}'.format(value['Parameters']['Kappa'])
            color = 'green'
        else:
            label = 'Scénario constant avec $log_{10}(κ) = $' \
                + '{}'.format(value['Parameters']['Kappa'])
            color= 'blue'
            
        # Plot
        ax.plot(normalized_sfs, label=label, color=color)
        
    # Plot theoretical SFS
    theoretical_sfs = normalization(compute_theoretical_sfs(len(normalized_sfs)))
    ax.plot(theoretical_sfs, color='orange', marker="o",
            label="SFS théorique constant - Fu, 1995")
    
    # Compute X axis ticks & values
    x_ax, x_values = [], []
    for i in range(10):
        x_ax.append(i+i)
        x_values.append("{}/{}".format(i*2+1, len(theoretical_sfs)+1))
        
    # X axis ticks & labels
    ax.set_xticks(x_ax)
    ax.set_xticklabels(x_values)

    # Label axis
    if xaxis:
        ax.set_xlabel("Fréquence allélique", fontsize="x-large")
    ax.set_ylabel("SNPs - pourcentage", fontsize="x-large")

    # Title
    ax.set_title(title, fontsize='xx-large')


def plot_all_sfs(df, suptitle, sfs, filout):
    """
    Plot the theoretical SFS vs the observed ones.
    
    Parameters
    ----------
    sfs: either observed or inferred
    """
    # Set up plot
    _,  ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
    ax4.set_visible(False)
    plt.subplots_adjust(hspace=0.16)
    
    # SFS to select fo each ax
    data = get_data(df, sfs)
    
    key = list(data.keys())
    title = "Scénario de changement de taille de population pour $log_{10}(τ) = $"
    plot_sfs(data[key[0]], ax1, title=title + "{}".format(key[0]), xaxis=False)
    plot_sfs(data[key[1]], ax2, title=title + "{}".format(key[1]))
    plot_sfs(data[key[2]], ax3, title=title + "{}".format(key[2]))

    # Sup title
    plt.suptitle(suptitle, fontsize='xx-large', fontweight='bold', y=0.936)

    # Common legend
    lines, labels = ax1.get_legend_handles_labels()
    #plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0. )
    plt.figlegend(lines, labels, fontsize='x-large', loc='center right', ncol=1,
                  bbox_to_anchor=(0.18, 0., 0.52, 0.4))
    
    plt.savefig(filout, format='png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    sys.exit()  # No actions desired
