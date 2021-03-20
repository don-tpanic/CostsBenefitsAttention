import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def convert_probs_to_topX_rates(exp_num, importance, task, random_seed):
    """
    Given a threshold, such as top5,
    for each class' pos probs and neg probs,
    grab the top5 probs, can compute true pos
    and false pos rates.
    """
    df = pd.read_csv('groupings-csv/imagenetA_Imagenet.csv', usecols=['wnid', 'idx', 'description'])
    sorted_indices = np.argsort([i for i in df['wnid']])
    group_classes = np.array([i for i in df['wnid']])[sorted_indices]
    group_indices = np.array([int(i) for i in df['idx']])[sorted_indices]
    group_descriptions = np.array([i for i in df['description']])[sorted_indices]

    if task == 'EXP':
        topX_dir = f'topX{random_seed}'
        probs_dir = f'probs{random_seed}'

    elif task == 'retrain':
        topX_dir = f'topX_retrain{random_seed}'
        probs_dir = f'probs_retrain{random_seed}'

    # go thru all thresholds
    for top_x in range(1,2):
        # check if this top_x has been done,
        topX_path = f'exp_results/cobb/exp{exp_num}/{topX_dir}'
        if not os.path.exists(topX_path):
            os.mkdir(topX_path)
        if os.path.exists(os.path.join(topX_path, f'top{top_x}_for_roc_imp{importance}.npy')):
            print(f'--- skipping [{top_x}] ---')
            continue
        else:
            # first col: true pos rate for all classes
            # second col: false pos rate for all classses
            all_true_n_false_pos_rates = np.zeros((len(group_classes), 2))
            # each model at a time
            for i in range(len(group_classes)):
                wnid = [group_classes[i]]  #wnid
                index = group_indices[i]  #network index aka true_label
                description = group_descriptions[i]  #description

                # (n, 1000)
                probs_pos = np.load(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{description}_imp{importance}_truePos.npy')
                probs_neg = np.load(f'exp_results/cobb/exp{exp_num}/{probs_dir}/{description}_imp{importance}_falsePos.npy')

                true_pos = 0
                false_pos = 0
                # one image at a time
                for row_i in range(probs_pos.shape[0]):
                    # if true label of the image is in top_x (true pos + 1)
                    top_x_indices_pos = np.argsort(probs_pos[row_i])[::-1][:top_x]
                    if index in top_x_indices_pos:
                        true_pos += 1

                #for row_i in range(probs_neg.shape[0]):
                #     # if true label of the image is in top_x (false pos + 1)
                    top_x_indices_neg = np.argsort(probs_neg[row_i])[::-1][:top_x]
                    if index in top_x_indices_neg:
                        false_pos += 1

                # rate per class (scalar)
                true_pos_rate = true_pos / probs_pos.shape[0]
                false_pos_rate = false_pos / probs_neg.shape[0]
                # for each top_x, we collect all 200 pairs
                #print(i)
                all_true_n_false_pos_rates[i, :] = [true_pos_rate, false_pos_rate]
            
            # save as (200,2) for each top_x
            np.save(os.path.join(topX_path, f'top{top_x}_for_roc_imp{importance}.npy'), all_true_n_false_pos_rates)
            print(f'** exp=[{exp_num}], top=[{top_x}], imp=[{importance}] saved **')


def convert_probs_to_topX_rates_EXP2(exp_num, importance, task, random_seed):
    df = pd.read_csv('groupings-csv/imagenetA_Imagenet.csv', usecols=['wnid', 'idx', 'description'])
    sorted_indices = np.argsort([i for i in df['wnid']])
    group_classes = np.array([i for i in df['wnid']])[sorted_indices]
    group_indices = np.array([int(i) for i in df['idx']])[sorted_indices]
    group_descriptions = np.array([i for i in df['description']])[sorted_indices]

    if task == 'EXP':
        topX_dir_src = f'topX{random_seed}'
        probs_dir_src = f'probs{random_seed}'

        topX_dir_det = f'topX_v2{random_seed}'
        probs_dir_det = f'probs_v2{random_seed}'

    elif task == 'retrain':
        topX_dir_src = f'topX_retrain{random_seed}'
        probs_dir_src = f'probs_retrain{random_seed}'

        topX_dir_det = f'topX_retrain_v2{random_seed}'
        probs_dir_det = f'probs_retrain_v2{random_seed}'

    if os.path.exists(f'exp_results/cobb/exp{exp_num}/{probs_dir_det}'):
        print(f'{probs_dir_det} exists, skip to convert to topX.')
    else:
        # go thru all thresholds
        for top_x in range(1, 2):
            # check if this top_x has been done,
            probs_path = f'exp_results/cobb/exp{exp_num}/{probs_dir_det}'
            if not os.path.exists(probs_path):
                os.mkdir(probs_path)
            else:
                # first col: true pos rate for all classes
                # second col: false pos rate for all classses
                all_true_n_false_pos_rates = np.zeros((len(group_classes), 2))
                # each model at a time
                for i in range(len(group_classes)):
                    wnid = [group_classes[i]]  # wnid
                    index = group_indices[i]  # network index aka true_label
                    description = group_descriptions[i]  # description

                    probs_pos = np.load(f'exp_results/cobb/exp{exp_num}/{probs_dir_src}/{description}_imp{importance}_truePos.npy')
                    probs_neg = np.load(f'exp_results/cobb/exp{exp_num}/{probs_dir_src}/{description}_imp{importance}_falsePos.npy')
                    
                    # Then we group the ouptputs by pair of classes within an attn model,
                    # so the probs are assessed on per pair of blends basis, not across all pairs.
                    step_size = int(probs_pos.shape[0] / (len(group_classes) - 1))
                    for z in range(len(group_classes)-1):
                        first_index = step_size * z
                        second_index = step_size * z + step_size
                        print(z, first_index, second_index)
                        
                        probs_pos_per_pair = probs_pos[first_index:second_index, :]
                        probs_neg_per_pair = probs_neg[first_index:second_index, :]
                        np.save(f'exp_results/cobb/exp{exp_num}/{probs_dir_det}/{description}_imp{importance}_truePos_pair{z}.npy', probs_pos_per_pair)
                        np.save(f'exp_results/cobb/exp{exp_num}/{probs_dir_det}/{description}_imp{importance}_falsePos_pair{z}.npy', probs_neg_per_pair)

    print('Converting to topX...')
    # go thru all thresholds
    for top_x in range(1, 2):
        # check if this top_x has been done,
        topX_path = f'exp_results/cobb/exp{exp_num}/{topX_dir_det}'
        if not os.path.exists(topX_path):
            os.mkdir(topX_path)
        else:
            # first col: true pos rate for all classes
            # second col: false pos rate for all classses
            #all_true_n_false_pos_rates = np.zeros((len(group_classes), 2))
            all_true_n_false_pos_rates = []
            # each model at a time
            for i in range(len(group_classes)):
                wnid = [group_classes[i]]  #wnid
                index = group_indices[i]  #network index aka true_label
                description = group_descriptions[i]  #description

                # within each class, load a pair at a time.
                for z in range(len(group_classes)-1):
                    # (n, 1000)
                    probs_pos_per_pair = np.load(f'exp_results/cobb/exp{exp_num}/{probs_dir_det}/{description}_imp{importance}_truePos_pair{z}.npy')
                    probs_neg_per_pair = np.load(f'exp_results/cobb/exp{exp_num}/{probs_dir_det}/{description}_imp{importance}_falsePos_pair{z}.npy')                           
                
                    true_pos = 0
                    false_pos = 0
                    # one image at a time
                    for row_i in range(probs_pos_per_pair.shape[0]):
                        # if true label of the image is in top_x (true pos + 1)
                        top_x_indices_pos = np.argsort(probs_pos_per_pair[row_i])[::-1][:top_x]
                        if index in top_x_indices_pos:
                            true_pos += 1

                        # if true label of the image is in top_x (false pos + 1)
                        top_x_indices_neg = np.argsort(probs_neg_per_pair[row_i])[::-1][:top_x]
                        if index in top_x_indices_neg:
                            false_pos += 1

                    # rate per class (scalar)
                    true_pos_rate = true_pos / probs_pos_per_pair.shape[0]
                    false_pos_rate = false_pos / probs_neg_per_pair.shape[0]
                    #all_true_n_false_pos_rates[i, :] = [true_pos_rate, false_pos_rate]
                    all_true_n_false_pos_rates.append([true_pos_rate, false_pos_rate])
            
            # save as (200,2) for each top_x
            np.save(os.path.join(topX_path, f'top{top_x}_for_roc_imp{importance}.npy'), all_true_n_false_pos_rates)
            print(f'** top=[{top_x}], imp=[{importance}] saved **')


def ad_hoc_SDT(hit_n_fas):
    """
    Compute one dprime and criterion for one set of hit and fas rates.
    MacMillain correction is used (Stanislaw & Todorov, 1999)

    inputs:
    ------
        hit_n_fas: is a (200,2) matrix
    """
    d_primes = []
    criterions = []

    for i in range(hit_n_fas.shape[0]):
        hit_rate = hit_n_fas[i, 0]
        fas_rate = hit_n_fas[i, 1]

        # macmillain correction.
        correction = 0.5 / hit_n_fas.shape[0]
        if hit_rate == 1:
            hit_rate = 1 - correction
        if hit_rate == 0:
            hit_rate = correction

        if fas_rate == 1:
            fas_rate = 1 - correction
        if fas_rate == 0:
            fas_rate = correction

        d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fas_rate)
        c = -(stats.norm.ppf(hit_rate) + stats.norm.ppf(fas_rate)) / 2
        d_primes.append(d_prime)
        criterions.append(c)

    return d_primes, criterions


def plot_EXP_all_metrics(exp_num, random_seed, top_x=1):
    """
    Plot top1 violins for all exp1-3 ICML results.
    """
    #from matplotlib import rc
    #rc('text', usetex=True)
    if exp_num == '1' or exp_num == '3':
        topX_path = f'exp_results/cobb/exp{exp_num}/topX{random_seed}'
    else:
        topX_path = f'exp_results/cobb/exp{exp_num}/topX_v2{random_seed}'

    importances = [1, 0.5, round(1/999, 3), 1/999000, 0]
    tests = {'hits': 0, 'fas': 1, 'd_prime': 2, 'c': 3}

    fig, ax = plt.subplots(1,len(tests), figsize=(15,6))

    for test in tests:
        data = []  # all hits or fas
        for i in range(len(importances)):
            importance = importances[i]

            # (200, 2)
            hit_n_fas_file = os.path.join(topX_path, f'top{top_x}_for_roc_imp{importance}.npy')
            hit_n_fas = np.load(hit_n_fas_file)
            if test == 'hits':
                ax[tests[test]].set_title('Hit Rates')
                res = hit_n_fas[:, 0]
            if test == 'fas':
                ax[tests[test]].set_title('False Alarm Rates')
                res = hit_n_fas[:, 1]
            if test == 'd_prime':
                ax[tests[test]].set_title(r'$d^\prime$')
                res, _ = ad_hoc_SDT(hit_n_fas)
            if test == 'c':
                ax[tests[test]].set_title(r'Criterion')
                _, res = ad_hoc_SDT(hit_n_fas)
            print(f'CHECK: test = {test}, len(res) = ', len(res))
            data.append(res)

        plots = ax[tests[test]].violinplot(dataset=data, positions=range(1, len(importances)+1), points=200, widths=0.5, showmeans=False, showextrema=False, showmedians=False)
        colors = ['grey'] * len(importances)
        for j in range(len(plots['bodies'])):
            pc = plots['bodies'][j]
            pc.set_facecolor(colors[j])

        quartile1 = []
        medians = []
        quartile3 = []
        for d in data:
            q1, md, q3 = np.percentile(d, [25,50,75])
            quartile1.append(q1)
            medians.append(md)
            quartile3.append(q3)

        inds = np.arange(1, len(medians) + 1)
        ax[tests[test]].scatter(inds, medians, marker='o', color='red', s=30, zorder=3)
        ax[tests[test]].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=2)

        ax[tests[test]].set_xticks(range(1, len(importances)+1))
        ax[tests[test]].set_xticklabels([r"$\alpha=0.001$", r"$\alpha=0.002$", r'$\alpha=0.5$', r'$\alpha=0.999$', r'$\alpha=1$'], rotation=56)
        ax[tests[test]].grid(which='major', axis='y', linestyle='--')

    fig.tight_layout()
    plt.savefig(f'exp_results/cobb/exp{exp_num}_top{top_x}.pdf')
    print('plotted.')

    
def plot_RETRAIN_all_metrics(exp_num, random_seed, top_x=1):
    """
    Plot retraining results.
    """
    tests = {'hits': 0, 'fas': 1, 'd_prime': 2, 'c': 3}
    fig, ax = plt.subplots(1, len(tests), figsize=(15,6))
    versions = ['attn', 'retrain']

    for test in tests:
        data = []  # all hits or fas
        for v in versions:

            if v == 'retrain':
                if exp_num == '1' or exp_num == '3':
                    topX_path = f'exp_results/cobb/exp{exp_num}/topX_retrain{random_seed}'
                else:
                    topX_path = f'exp_results/cobb/exp{exp_num}/topX_retrain_v2{random_seed}'

            elif v == 'attn':
                if exp_num == '1' or exp_num == '3':
                    topX_path = f'exp_results/cobb/exp{exp_num}/topX{random_seed}'
                else:
                    topX_path = f'exp_results/cobb/exp{exp_num}/topX_v2{random_seed}'

            # (200, 2)
            hit_n_fas_file = os.path.join(topX_path, f'top{top_x}_for_roc_imp0.001.npy')
            hit_n_fas = np.load(hit_n_fas_file)

            if test == 'hits':
                ax[tests[test]].set_title('Hit Rates')
                res = hit_n_fas[:, 0]
            if test == 'fas':
                ax[tests[test]].set_title('False Alarm Rates')
                res = hit_n_fas[:, 1]
            if test == 'd_prime':
                ax[tests[test]].set_title(r'$d^\prime$')
                res, _ = ad_hoc_SDT(hit_n_fas)
            if test == 'c':
                ax[tests[test]].set_title(r'Criterion')
                _, res = ad_hoc_SDT(hit_n_fas)
            print(f'CHECK: test = {test}, len(res) = ', len(res))
            data.append(res)

        plots = ax[tests[test]].violinplot(dataset=data, positions=range(1, len(versions)+1), points=200, widths=0.5, showmeans=False, showextrema=False, showmedians=False)
        colors = ['grey'] * len(versions)
        for j in range(len(plots['bodies'])):
            pc = plots['bodies'][j]
            pc.set_facecolor(colors[j])

        quartile1 = []
        medians = []
        quartile3 = []
        for d in data:
            q1, md, q3 = np.percentile(d, [25,50,75])
            quartile1.append(q1)
            medians.append(md)
            quartile3.append(q3)

        inds = np.arange(1, len(medians) + 1)
        ax[tests[test]].scatter(inds, medians, marker='o', color='red', s=30, zorder=3)
        ax[tests[test]].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=2)

        ax[tests[test]].set_xticks(range(1, len(versions)+1))
        ax[tests[test]].set_xticklabels(['Attention', 'Retraining'])
        ax[tests[test]].grid(which='major', axis='y', linestyle='--')

    fig.tight_layout()
    if exp_num == '1':
        txt='(a) Experiment on Regular Images'
    elif exp_num == '2':
        txt='(b) Experiment on Blended Images'
    elif exp_num == '3':
        txt='(c) Experiment on Adversarial Images'

    fig.text(.5, .005, txt, ha='center', fontsize=18)
    plt.savefig(f'exp_results/cobb/exp{exp_num}_retrain-top{top_x}.pdf', bbox_inches='tight')
    print('plotted.')


def plot_topX_hit_rate(exp_num, random_seed):
    fig, ax = plt.subplots()
    topX_path = f'exp_results/cobb/exp{exp_num}/topX{random_seed}'
    top_x_range = np.arange(1, 1001)[::-1]

    # plot uniform and balanced attention on one roc.
    for imp in [1, 0.001]:
        hit_rates = []
        for top_x in top_x_range:
            # (200, 2) per threshold, all classes true pos and false pos rates.
            top_x_true_n_false_pos_rates = np.load(os.path.join(topX_path, f'top{top_x}_for_roc_imp{imp}.npy'))
            average_true_pos_rate = np.mean(top_x_true_n_false_pos_rates[:, 0])
            hit_rates.append(average_true_pos_rate)
        if imp == 1:
            label = 'uniform attention'
        if imp == 0.001:
            label = 'balanced attention'
        ax.scatter(top_x_range, hit_rates, label=label, alpha=0.3, s=plt.rcParams['lines.markersize'] ** 2 / 16)

    ax.invert_xaxis()
    temp = np.arange(0,1001,200)
    temp[0] = 1
    plt.xticks(np.arange(1,1002,200), temp)
    plt.xlabel('top x')
    plt.ylabel('correct reponse %')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'exp_results/cobb/exp{exp_num}_topX_hit_rate-average.pdf')


def execute(exp_num, task, converting=False, plotting=True):
    # exp_num = 1
    # task = 'EXP'
    random_seed = 42

    if converting:
        importances = [1, 0.5, 0.001, 1/999000, 0]
        
        for importance in importances:
            if exp_num == '1' or exp_num == '3':
                convert_probs_to_topX_rates(exp_num=exp_num, importance=importance, task=task, random_seed=random_seed)
            elif exp_num == '2':
                convert_probs_to_topX_rates_EXP2(exp_num=2, importance=importance, task=task, random_seed=random_seed)
    if plotting:
        if task == 'EXP':
            plot_EXP_all_metrics(exp_num, random_seed)
        elif task == 'retrain':
            plot_RETRAIN_all_metrics(exp_num, random_seed)
    
    
