import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from pandas import Index
from tensorflow.python.keras.utils.generic_utils import Progbar

from utils.metrics import calculate_mean_aoc_curve, load_morf_curves, calculate_mean_aoc_from_curves, load_metrics
from utils.misc import pretty_method_name, method_color, pretty_header_name, multiplication_by_x


def run_evaluate_bulk(methods, dataset_id, model_id, morfnum=250, baseline_method='random_uniform', figdim=5, group_name='1', sort_by=None, noplot=False, **kwargs):
    # Prepare placeholder
    results_list = []
    results_dict = {m: {} for m in methods}

    # Derive paths
    path_plots = 'plots/{}_{}'.format(dataset_id, model_id)
    path_tables = 'tables/{}_{}'.format(dataset_id, model_id)

    # Create directories for plots and tables if absent
    os.makedirs(path_plots, exist_ok=True)
    os.makedirs(path_tables, exist_ok=True)

    # Evaluate MoRF AOC values
    print('Evaluating MoRF AOC values...')
    results_list.append(evaluate_morf_aoc(methods, dataset_id, model_id, path_plots, morfnum=morfnum, baseline_method=baseline_method, figdim=figdim, group_name=group_name, noplot=noplot, **kwargs))

    # Evaluate other metrics
    print('Evaluating PCC values...')
    results_list.append(evaluate_metric(methods, dataset_id, model_id, metric='pcc', dround=2))

    print('Evaluating SCC values...')
    results_list.append(evaluate_metric(methods, dataset_id, model_id, metric='scc', dround=2))

    # Combine results to dict
    print('Combining results...')
    for m in methods:
        for r in results_list:
            results_dict[m].update(r[m])

    # Convert dict to dataframe
    df = pd.DataFrame.from_dict(results_dict, orient='index')

    # Sort dataframe
    if sort_by is not None:
        df = df.sort_values(by=sort_by, ascending=False)

    # Reformat dataframe: header
    header = [[], []]
    for c in df.columns:
        h1, h0 = str(c).rsplit('_', maxsplit=1)
        header[0].append(str(h0).upper())
        header[1].append(h1)
    df.columns = header

    # Reformat dataframe: index
    df.index = Index([pretty_method_name(m) for m in df.index], name='method')
    pd.set_option('display.max_colwidth', None)

    # Export dataframe to LaTeX
    df.to_latex(caption='Results for model {} on the {} dataset for selected methods.'.format(model_id, dataset_id),
                label='tab:{}_{}_g{}'.format(dataset_id, model_id, group_name),
                escape=False,
                multicolumn_format='c',
                buf='{}/table_results_g{}.tex'.format(path_tables, group_name))

    # df.to_excel('{}/table_results_g{}.xlsx'.format(path_tables, group_name))


def run_evaluate_bulk_multiple(methods, dataset_id, model_ids, metric, morfnum=250, group_name='1', aggr=['mean'], dround=2, colgroups=None, suffix_merge=None, **kwargs):
    # Prepare placeholder
    results_dict = {mid: {} for mid in model_ids}
    table_dict = {}

    # Derive paths
    path_tables = 'tables/{}'.format(dataset_id)

    # Create directories for plots and tables if absent
    os.makedirs(path_tables, exist_ok=True)

    for model_id in model_ids:
        # Prepare placeholder
        results_list_tmp = []
        results_dict_tmp = {m: {} for m in methods}

        if metric == 'aoc':
            # Evaluate MoRF AOC values
            print('Evaluating MoRF AOC values...')
            results_list_tmp.append(evaluate_morf_aoc(methods, dataset_id, model_id, morfnum=morfnum, path_plots=None, noplot=True, **kwargs))

        else:
            # Evaluate other metric
            print('Evaluating {} values...'.format(str(metric).upper()))
            results_list_tmp.append(evaluate_metric(methods, dataset_id, model_id, metric=metric, dround=dround))

        # Combine results to dict
        print('Combining results...')
        for m in methods:
            for r in results_list_tmp:
                results_dict_tmp[m].update(r[m])

        results_dict[model_id] = results_dict_tmp

    # Convert and condense elements to table structure
    for model_id in results_dict:
        for method in results_dict[model_id]:
            cellvalue = results_dict[model_id][method]['mean_{}'.format(metric)]

            # Perform suffix merge
            method_d = method

            if suffix_merge is not None:
                for r in suffix_merge:
                    for v in suffix_merge[r]:
                        if str(method).endswith(v):
                            method_d = str(method).replace(v, r)
                            break

            if cellvalue is not None:
                if pretty_method_name(method_d) not in table_dict.keys():
                    table_dict[pretty_method_name(method_d)] = {}
                table_dict[pretty_method_name(method_d)][model_id] = cellvalue

    # Convert dict to dataframe
    df = pd.DataFrame.from_dict(table_dict, orient='index')

    # Add aggregation column(s)
    for a in aggr:
        if a == 'mean':
            df[a] = df.mean(axis=1, skipna=True)
        elif a == 'min':
            df[a] = df.min(axis=1, skipna=True)
        elif a == 'max':
            df[a] = df.max(axis=1, skipna=True)
        if a == 'std':
            df[a] = df.std(axis=1, skipna=True)

    # Replace NaNs
    df = df.replace({np.nan: '-'})

    # Reformat dataframe: split header into groups
    if colgroups is not None:
        header = [[] for _ in colgroups]
        for c in df.columns:
            for i, g in enumerate(colgroups.keys()):
                for gv in colgroups[g]:
                    if type(gv) is not bool:
                        if gv in str(c):
                            header[i].append(pretty_header_name(gv))
                            break
                    else:
                        if g in str(c):
                            header[i].append(pretty_header_name('{}_{}'.format(g, gv)))
                else:
                    continue

        # Add aggregation columns to header and placeholders to header groups
        for a in aggr:
            header[0].append(a)
        for l in header[1:]:
            for _ in aggr:
                l.append('~')

        # Set header
        df.columns = header

    else:
        df.columns = [pretty_header_name(h) for h in list(df.columns)]

    # Reformat dataframe: index
    pd.set_option('display.max_colwidth', None)

    # Export dataframe to LaTeX
    df.to_latex(caption='Comparison of different models trained on {} based on metric {}'.format(dataset_id, metric.upper()),
                label='tab:{}_model_comparison_{}'.format(dataset_id, metric),
                escape=False,
                column_format='l'+'r'*len(df.columns),
                multicolumn_format='c',
                float_format=str('{:0.' + str(dround) + 'f}').format,
                buf='{}/table_{}_model_comparison_{}_g{}.tex'.format(path_tables, dataset_id, metric.upper(), group_name))


def evaluate_metric(methods, dataset_id, model_id, metric, dround=4):
    # Prepare placeholder
    return_values = {}

    # Iterate over methods
    for m in methods:
        try:
            # Load metric values for all samples in dataset
            values = load_metrics(dataset_id, model_id, m, metric)
            values = np.nan_to_num(values)

            # Calculate mean and std of AOC
            return_values[m] = {'mean_{}'.format(metric): round(float(np.mean(values)), dround), 'std_{}'.format(metric): round(float(np.std(values)), dround)}

        except FileNotFoundError:
            # print('Skipping {} evaluation for "{}" as file is non-existent.'.format(metric.upper(), m))
            return_values[m] = {'mean_{}'.format(metric): None, 'std_{}'.format(metric): None}

    return return_values


def evaluate_morf_aoc(methods, dataset_id, model_id, path_plots, morfnum=250, baseline_method='random_uniform', figdim=5, group_name='1', noplot=False, legend_inside=False, **kwargs):
    # Prepare placeholders
    all_results = []
    return_values = {}

    # Calculcate mean morf curve and mean aoc curve for baseline method
    morf_curves_baseline = load_morf_curves(dataset_id, model_id, baseline_method)
    mean_aoc_curve_baseline, mean_morf_curve_baseline = calculate_mean_aoc_curve(morf_curves_baseline, return_mean_morf_curve=True)

    # Define progressbar
    pb = Progbar(len(methods), width=50, verbose=1, interval=0.05)

    # Iterate over methods
    for i, m in enumerate(methods):
        # Update progressbar
        pb.update(i+1)

        try:
            # Load morf curves for all samples in dataset
            morf_curves = load_morf_curves(dataset_id, model_id, m)

            # Calculate mean and std of AOC
            mean_aoc, std_aoc = calculate_mean_aoc_from_curves(morf_curves)
            return_values[m] = {'mean_aoc': round(mean_aoc, 4), 'std_aoc': round(std_aoc, 4)}

            # Calculate mean morf curve and return AOC at each position = AOC curve
            mean_aoc_curve, mean_morf_curve = calculate_mean_aoc_curve(morf_curves, return_mean_morf_curve=True)

            # Calculate curves relative to baseline
            aoc_curve_relative_to_baseline = ((mean_aoc_curve / mean_aoc_curve_baseline) * 100) - 100
            morf_curve_relative_to_baseline = ((mean_morf_curve / mean_morf_curve_baseline) * 100) - 100

            # AOC
            if m != baseline_method:
                a = mean_aoc_curve[-1]
            else:
                a = 0

            # Keep AOC, AOC curve and AOC curve relative to baseline for plots below
            all_results.append((a, mean_aoc_curve, aoc_curve_relative_to_baseline, mean_morf_curve, morf_curve_relative_to_baseline, m))

        except FileNotFoundError:
            print('\nSkipping MoRF AOC evaluation for "{}" as file is non-existent.\n'.format(m))
            return_values[m] = {'mean_aoc': None, 'std_aoc': None}

    if not noplot:
        # Plot mean MoRF curves
        if legend_inside:
            plt.figure(figsize=(figdim * 1.2, figdim))
        else:
            plt.figure(figsize=(figdim * 1.8, figdim*1.05))

        for i, (a, mc, bc, mmc, mmbc, m) in enumerate(sorted(all_results).__reversed__()):
            if m == baseline_method:
                c = 'k'
            else:
                c = method_color(m)
            plt.plot(mmc, c, label='{}'.format(pretty_method_name(m, wo_params=True)))

        ticks = np.linspace(start=0, stop=morfnum, num=5)
        labels = [r'${}$'.format(x) for x in np.array(ticks / morfnum * 100, dtype=int)]
        plt.xticks(ticks=ticks, labels=labels)
        plt.xlabel(r'perturbed area (\%)')
        plt.xlim((0, np.max(ticks)))
        plt.ylim((0, None))
        plt.ylabel(r'$f_c(x)$')

        if legend_inside:
            plt.legend()
        else:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig('{}/morf_curves_mean_g{}'.format(path_plots, group_name))
        plt.close()

    return return_values


def run_aoc_corr_plot(confs, metric='pcc', morfnum=250, baseline_method='random_uniform', figdim=10, show_labels=True, adjust=True, **kwargs):
    if show_labels is False:
        figdim = figdim * 0.55
        s = 20
    else:
        s = 30

    results = []

    # Derive path
    path_plots = 'plots/AOC_{}'.format(metric.upper())

    # Create directory for plots if absent
    os.makedirs(path_plots, exist_ok=True)

    for conf in confs:
        dataset_id = conf['dataset_id']
        model_ids = conf['model_ids']
        methods = conf['methods']

        for model_id in model_ids:
            # Evaluate MoRF AOC values
            print('Evaluating MoRF AOC values...')
            tmp_aocs = evaluate_morf_aoc(methods, dataset_id, model_id, path_plots, morfnum=morfnum, baseline_method=baseline_method, figdim=figdim, noplot=True, **kwargs)

            # Evaluate other metrics
            print('Evaluating {} values...'.format(metric.upper()))
            tmp_corr_values = evaluate_metric(methods, dataset_id, model_id, metric=metric, dround=2)

            for m in methods:
                if (tmp_aocs[m]['mean_aoc'] is not None) and (tmp_corr_values[m]['mean_{}'.format(metric)] is not None):
                    results.append({'dataset_id': dataset_id, 'model_id': model_id, 'method': m, 'mean_aoc': tmp_aocs[m]['mean_aoc'], 'mean_corr': tmp_corr_values[m]['mean_{}'.format(metric)]})

    # Convert dict to dataframe
    df = pd.DataFrame(results)
    # df.to_excel('{}/AOC_vs_{}_tmp.xlsx'.format(path_plots, metric.upper()), index=False)
    # df = pd.read_excel('{}/AOC_vs_{}_tmp.xlsx'.format(path_plots, metric.upper()))

    # Sort dataframe by method
    df = df.sort_values('method')

    # Derive times_x column based on methods
    df['times_x'] = [multiplication_by_x(m) for m in df.method.values]

    for dataset_id in set(list(df.dataset_id.values) + [None]):
        print(dataset_id)

        # Filter dataframe if dataset_id was given
        if dataset_id is not None:
            df_tmp = df[df.dataset_id == dataset_id]
        else:
            df_tmp = df

        # Derive color values
        color_map = {2: 'dodgerblue', 1: 'red', 0: 'black'}
        colors = [color_map[t] for t in df_tmp.times_x.values]

        # Counter and collector variables
        texts = []
        legend_counter = {0: -int(len(df_tmp) * 0.3), 1: 0, 2: 0}

        # Begin figure
        plt.figure(figsize=(figdim, figdim))

        # Plot
        for x, y, c, t, m in zip(df_tmp.mean_corr.values, df_tmp.mean_aoc.values, colors, df_tmp.times_x.values, df_tmp.method.values):
            # Add only first point of each color to legend
            if legend_counter[t] == 0:
                label = {0: 'other', 1: r'$\times$ Input', 2: r'$\times$ SIGN'}[t]
            else:
                label = '_nolegend'

            # Plot points
            plt.scatter(x, y, color=c, label=label, s=s, alpha=0.5)

            # Increase legend counter
            legend_counter[t] += 1

        # Add labels
        if show_labels:
            # Sort dataframe by AOC, to label from bottom to top
            df_tmp = df_tmp.sort_values('mean_aoc')
            for x, y, m, i in zip(df_tmp.mean_corr.values, df_tmp.mean_aoc.values, df_tmp.method.values, range(len(df_tmp))):
                # If dataset_id given, annotate all points. If all datasets are plotted, annotate only in less dense areas or use 1/3 of points
                if (dataset_id is not None) or (y < 0.85) or ((i % 3) == 0):
                    texts.append(plt.annotate(pretty_method_name(m, wo_params=True), (x, y), fontsize=7))

        # Add axis labels
        plt.xlabel(r'$\overline{\text{' + metric.upper() + r'}}$')
        plt.ylabel(r'$\overline{\text{AOC}}$')

        # Add legend
        plt.legend(loc='lower right')

        # Make layout tight
        plt.tight_layout()

        # Adjust label positions if necessary
        if show_labels and adjust:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

        # Save figure
        path_fig = '{}/AOC_vs_{}_{}_{}.pdf'.format(path_plots, metric.upper(), dataset_id, {True: 'w_labels', False: 'no_labels'}[show_labels])
        plt.savefig(path_fig)
        print('Saved "{}"'.format(path_fig))
        plt.close()

