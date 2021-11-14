import matplotlib
import matplotlib.pyplot as plt
import os, fnmatch
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
import argparse


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 18
sign_pair = ['<', '>=']
inequality_operators = {'<': lambda x, y: x < y,
                        '<=': lambda x, y: x <= y,
                        '>': lambda x, y: x > y,
                        '>=': lambda x, y: x >= y}


plt.rcParams.update({'font.size': BIGGER_SIZE})


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result




# p_val_cont = fdr_df.loc[fdr_df.apply(lambda x: '(Cont.)' in x['outcome'], axis=1), 'p_val'].values
# p_val_bin = fdr_df.loc[fdr_df.apply(lambda x: '(Binary)' in x['outcome'], axis=1), 'p_val'].values
#
# _, fdr_cont = fdrcorrection(p_val_cont)
# _, fdr_bin = fdrcorrection(p_val_bin)
#
# fdr_df.loc[fdr_df.apply(lambda x: '(Cont.)' in x['outcome'], axis=1), 'fdr'] = fdr_cont
# fdr_df.loc[fdr_df.apply(lambda x: '(Binary)' in x['outcome'], axis=1), 'fdr'] = fdr_bin
relation_dict = {'pos_correlate': 2.0,
                 # 'mixed_sign_profile': 0.0,
                 'neg_correlate': 1.0,
                 # 'na': 0
                 }


# relation_dict = {'Positive Coef.': 2.0,
#                  # 'mixed_sign_profile': 0.0,
#                  'Negative Coef.': 1.0,
#                  # 'na': 0
#                  }

# relation_dict_inv = {v:k for k, v in relation_dict.items()}
relation_list = [k for k in relation_dict]
# relation_list = ['NA'] + relation_list
# relation_list.append('NA')
relation_list.append('NA')

relation_inv_dict = {v: k for k, v in relation_dict.items()}


def summarize_plot(result_dir='', pollutant_suffix='', method_suffix=''):
    # print(pollutant_suffix)



    fdr_path = os.path.join(result_dir, 'merged_fdr.csv')
    fdr_df = pd.read_csv(fdr_path, sep=',')
    print(fdr_df)

    fdr_df = fdr_df.loc[fdr_df['fdr'] < 0.05]
    # fdr_df = fdr_df.loc[fdr_df['relation'] == 'pos_correlate']
    print(fdr_df)

    # fdr_df['profile'] = fdr_df['profile'].str.replace(pollutant_suffix, '', regex=False)

    profile_str = fdr_df['profile'].str
    print(profile_str)
    print((profile_str.count('>') > 1) & (profile_str.count('<') == 0))
    # print(profile_str)

    fdr_df['all_greater'] = (profile_str.count('>') > 1) & (profile_str.count('<') == 0)
    fdr_df['all_less'] = (profile_str.count('<') > 1) & (profile_str.count('>') == 0)
    # merged_df['multi_pollutants'] = False
    fdr_df['multi_pollutants'] = profile_str.count('\t\t') > 0
    fdr_df = fdr_df.rename(columns={'coef': 'mean'})
    print(fdr_df.columns)

    category_outcome = {'all_greater_combination': fdr_df['all_greater'],
                        # 'all_less': merged_df['all_less'],
                        # 'mixed_sign_multi_pollutants': (merged_df['multi_pollutants'] & ~merged_df['all_greater'] & ~merged_df['all_less']),
                        'individual_pollutant': ~fdr_df['multi_pollutants']}


    for profile_cat, profile_bool in category_outcome.items():
        fdr_df_cat = fdr_df[profile_bool]

        fdr_df_cat['se'] = np.abs((fdr_df_cat['mean'] - fdr_df_cat['coef_95CI_lower']))/1.96
        fdr_df_cat['mean'] = np.abs(fdr_df_cat['mean'])

        fdr_df_cat.sort_values(by=['outcome', 'mean'],
                                             ascending=[True, False],
                                             inplace=True)
        fdr_df_cat.reset_index(inplace=True)
        fdr_df_cat['Modelnum'] = fdr_df_cat.index
        fdr_df_cat = fdr_df_cat[['Modelnum', 'mean', 'se', 'freq', 'fdr', 'outcome', 'profile']]
        fdr_df_cat['pol1'] = np.nan
        fdr_df_cat['fdr_str'] = ''
        for row_idx, pollutant_row in fdr_df_cat.iterrows():
            p = pollutant_row['profile'].split('\t\t')
            for p_idx, p_sub in enumerate(p):
                p_coln = 'pol{}'.format(p_idx+1)
                if not (p_coln in fdr_df_cat.columns):
                    fdr_df_cat[p_coln] = np.nan
                p_no_sign = p[p_idx].split(sign_pair[1])[0].split(sign_pair[0])[0].title().split('(')[0]
                fdr_df_cat.loc[row_idx,p_coln] = p_no_sign
            p_fdr = fdr_df_cat.loc[row_idx, 'fdr']
            if p_fdr >= 0.01:
                fdr_df_cat.loc[row_idx, 'fdr_str'] = '{:.2f}'.format(p_fdr)
            else:
                fdr_df_cat.loc[row_idx, 'fdr_str'] = '<0.01'
        fdr_df_cat.drop(columns=['fdr'], inplace=True)
        fdr_df_cat.rename(columns={'fdr_str': 'fdr'}, inplace=True)
        fdr_df_cat['out_first_only'] = np.nan
        outcome_unique = fdr_df_cat['outcome'].unique
        for u_out in outcome_unique:
            first_idx = fdr_df_cat[fdr_df_cat['out'] == u_out].first_valid_index()
            fdr_df_cat.loc[first_idx, 'out_first_only'] = u_out
        fdr_df_cat.to_csv('{}/r_summary_{}.csv'.format(result_dir, profile_cat))

        r_cmd = "R CMD BATCH --no-save --no-restore '--args result_dir=\"{}\"' Rscript/profile_table.R".format(result_dir)
        os.system(r_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters of DEEP extraction')
    parser.add_argument('--result_dir', '-rd', type=str, required=True, help='Directory of DEEP result files')
    # outcome_fdr = find('fdr*.csv', )
    args = parser.parse_args()
    summarize_plot(result_dir=args.result_dir)