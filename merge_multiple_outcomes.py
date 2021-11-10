import sys
import os
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import argparse
import fnmatch

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # print(name)
            if fnmatch.fnmatch(name, pattern):
                # print('matched')
                result.append(name)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters of merging fdr files from multiple outcomes')
    parser.add_argument('--result_dir', '-rd', type=str, required=True, help='Directory of DEEP result files')

    # suffix = sys.argv[-1]
    args = parser.parse_args()
    result_dir = args.result_dir

    fdr_file_list = find('fdr*.csv', result_dir)

    p_val_list = []
    pred_score_list = []
    for fdr_file in fdr_file_list:
        p_val_list.append(pd.read_csv(os.path.join(result_dir,fdr_file)))

    pred_file_list = find('pred_score*.csv', result_dir)
    for pred_file in pred_file_list:
        pred_score_list.append(pd.read_csv(os.path.join(result_dir,pred_file)))

    pvalue_df_cat = pd.concat(p_val_list, ignore_index=True)
    pred_score_df_cat = pd.concat(pred_score_list, ignore_index=True)

    pvalue_df_cat['fdr'] = 0.0
    bool_list_fdr = [
                     (pvalue_df_cat['binary_outcome'] == True),
                     (pvalue_df_cat['binary_outcome'] == False)
                     ]
    for b in bool_list_fdr:
        pvalue_df_cat.loc[b,'fdr'] = fdrcorrection(pvalue_df_cat.loc[b,'p_val'].values)[-1]

    pvalue_df_cat.to_csv(os.path.join(result_dir, 'merged_fdr.csv'), index=False, header=True)
    pred_score_df_cat.to_csv(os.path.join(result_dir,'merged_pred_score.csv'), index=False, header=True)

