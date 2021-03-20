import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
plt.rcParams.update({'font.size': 16})

from matplotlib import rc
rc('text', usetex=True)


def execute():
    # alpha = [0, 0.5, 1-beta, 1-(beta/1000), 1]
    importances = [1, 0.5, 0.001, 1/999000, 0]
    alphas = [r"$\alpha=0$", r"$\alpha=0.5$", r'$\alpha=1-\beta$', r'$\alpha=1-(\beta/1000)$', r'$\alpha=1$']

    fig, ax = plt.subplots(1, len(importances), figsize=(14,4))

    for i in range(len(importances)):
        importance = importances[i]
        attention_dir = 'attention_weights/block4_pool/attention={imp}'.format(imp=importance)
        all_files = os.listdir(attention_dir)
        assert len(all_files) == 200, "wrong file number"

        tot_weights = []
        for f in all_files:
            weights = np.load('attention_weights/block4_pool/attention={imp}/'.format(imp=importance)+f)
            tot_weights.append(weights)

        tot_weights = np.array(tot_weights).flatten()

        if i >= 1:
            ax[i].set_yticks([])
        ax[i].set_xticks([0, 1, 4])
        ax[i].hist(tot_weights, density=True, bins=145)
        ax[0].set_ylabel('Probability Density', fontsize=18)
        ax[i].set_xlabel(alphas[i])
        ax[i].set_xlim([-0.2,4])
        ax[i].set_ylim([0,5.5])

    fig.text(0.5, 0.03, 'Attention Weights', ha='center', va='center', fontsize=18)
    plt.tight_layout(pad=2)
    plt.savefig('exp_results/cobb/attention_histogram.pdf')
