
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import scipy

from scipy import stats
import xgboost
import warnings
import argparse

from utils_DEEP import *
# from xgb_custom_visualizer import *
# import math

warnings.filterwarnings("ignore")
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
# plt.rcParams.update({'font.size': 12})
params = {'mathtext.default': 'regular'}
# plt.rcParams.update(params)

inequality_operators = {'<': lambda a, b: a < b,
                        '<=': lambda a, b: a <= b,
                        '>': lambda a, b: a > b,
                        '>=': lambda a, b: a >= b}

dataDir = os.path.join(os.getcwd(), 'data')  # default unless specified otherwise
tree_seed = 0

def runWorkflow(**kargs):
    verbose = kargs.get('verbose', True)
    input_file = kargs.get('input_file', '')
    binary_outcome = kargs.get('binary_outcome', True)
    outcome_folder_name = kargs.get('output_folder_name', '')
    p_val_df = kargs.get('p_val_df', pd.DataFrame({}))
    test_score_df = kargs.get('test_score_df', pd.DataFrame({}))
    # yr_name = kargs.get('yr_name', '')
    xgb = kargs.get('xgb', False)
    # outcome_name = kargs.get('outcome_name', pd.DataFrame({}))

    outputDir = os.path.join(plotDir, outcome_folder_name)
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    possible_results = ['pos_correlate', 'neg_correlate', 'mixed_sign_profile']
    possibleDirs = []
    for possible_result in possible_results:
        possibleDirs.append(os.path.join(outputDir, possible_result))
        if not os.path.exists(possibleDirs[-1]):
            os.mkdir(possibleDirs[-1])

    if verbose: print("(runWorkflow) 1. Specifying input data ...")
    ######################################################
    file_prefix = input_file.split('.')[0]

    confounding_vars = ['age', 'avg_income',
                        'race/ethnicity_Asian',
                        'race/ethnicity_Black or African American',
                        'race/ethnicity_Hispanic or Latino',
                        'race/ethnicity_More Than One Race',
                        'race/ethnicity_Unknown / Not Reported',
                        'race/ethnicity_White',
                        'gender',
                        ]

    exclude_vars = confounding_vars + ["ID"]

    """
    Load Data
    """

    X, y, features, confounders_df, whole_df = load_data(input_path=dataDir,
                                                         input_file=input_file,
                                                         exclude_vars=exclude_vars,
                                                         # col_target='Outcome',
                                                         confounding_vars=confounding_vars,
                                                         verbose=True)

    # 2. define model (e.g. decision tree)
    if verbose: print("(runWorkflow) 2. Define model (e.g. decision tree and its parameters) ...")
    ######################################################
    p_grid = {"min_samples_leaf": []}

    if xgb:
        model = xgboost.XGBClassifier(random_state=1)
        sign_pair = ['<', '>=']
    else:
        model = DecisionTreeClassifier(criterion='entropy', random_state=1)
        sign_pair = ['<=', '>']

    # 3. visualize the tree (deferred to analyze_path())
    ######################################################
    test_size = 0.2
    ######################################################
    labels = [str(l) for l in sorted(np.unique(y))]

    feature_idx_dict = {}
    for idx, feature in enumerate(features):
        feature_idx_dict[feature] = idx

    outcome_dir = os.path.join(plotDir, outcome_folder_name)

    """
    Stage 1 : Train 100 XGBoost Model and extract the decision paths and their frequencies
    """

    fmap_fn = features_to_txt(features)
    scores, list_params, topk_profile_str, sorted_paths, paths_median_threshold, visualize_dict = \
        analyze_path(X, y, model=model, p_grid=p_grid, feature_set=features, n_trials=100, n_trials_ms=30,
                     save=False,
                     merge_labels=False,
                     policy_count='standard',
                     experiment_id=file_prefix,
                     create_dir=True, index=0, validate_tree=False, to_str=True, verbose=False,
                     binary_outcome=binary_outcome,
                     fmap_fn=fmap_fn,
                     plot_dir=plotDir,
                     outcome_name=outcome_folder_name, xgb=xgb)

    scores_df = pd.DataFrame({'f_minority': scores['scores'][0],
                              'f_majority': scores['scores'][1],
                              'auc': scores['scores'][2],
                              })
    scores_df.to_csv(os.path.join(outcome_dir, 'scores_df.csv'))
    scores_rand_df = pd.DataFrame({
        'f_minority_rand': scores['scores_random'][0],
        'f_majority_rand': scores['scores_random'][1],
        'auc_rand': scores['scores_random'][2]
    })
    scores_rand_df.to_csv(os.path.join(outcome_dir, 'scores_rand_df.csv'))

    """
    Stage 2: Statistical assessment with potential confounders
    """
    p_val_df = statistical_assessment_with_confounder(sorted_paths, feature_idx_dict, paths_median_threshold,
                                           sign_pair, topk_profile_str, confounders_df,
                                           binary_outcome, y, visualize_dict, test_size,outcome_dir,fmap_fn,
                                         labels, X, possibleDirs, outcome_folder_name, file_prefix, outputDir, p_val_df)

    print('Finished All regressions!')

    # if len(p_val_df['outcome']==output_folder_name) == 0:
    if binary_outcome:
        f_minority = scores['scores'][0]
        f_minority_rand = scores['scores_random'][0]
        f_majority = scores['scores'][1]
        f_majority_rand = scores['scores_random'][1]
        auc = scores['scores'][2]
        auc_rand = scores['scores_random'][2]
        r2 = []
        r2_rand = []
    else:
        f_minority = []
        f_minority_rand = []
        f_majority = []
        f_majority_rand = []
        auc = []
        auc_rand = []
        r2 = scores['scores'][0]
        r2_rand = scores['scores_random'][0]

    if xgb:
        min_sl = 0
    else:
        min_sl = scipy.stats.mode(np.array(list_params['min_sample_leaf']))[0][0]
    test_cols = test_score_df.columns
    test_score_df = test_score_df.append({
        test_cols[0]: outcome_folder_name,
        test_cols[1]: binary_outcome,
        test_cols[2]: min_sl,
        # test_cols[3]: yr_name,
        'mean (std) r2 score from random predictors': '{:.3e}({:.3e})'.format(np.mean(r2_rand), np.std(r2_rand)),
        'mean (std) r2 score': '{:.3e}({:.3e})'.format(np.mean(r2), np.std(r2)),
        'mean (std) f score (minority) from random predictors': '{:.3e}({:.3e})'.format(np.mean(f_minority_rand),
                                                                                        np.std(f_minority_rand)),
        'mean (std) f score (minority)': '{:.3e}({:.3e})'.format(np.mean(f_minority), np.std(f_minority)),
        'mean (std) f score (majority) from random predictors': '{:.3e}({:.3e})'.format(np.mean(f_majority_rand),
                                                                                        np.std(f_majority_rand)),
        'mean (std) f score (majority)': '{:.3e}({:.3e})'.format(np.mean(f_majority), np.std(f_majority)),
        'mean (std) AUC score from random predictors': '{:.3e}({:.3e})'.format(np.mean(auc_rand), np.std(auc_rand)),
        'mean (std) AUC score': '{:.3e}({:.3e})'.format(np.mean(auc), np.std(auc)),
        test_cols[-1]: X.shape[0],
    }, ignore_index=True)

    return p_val_df, test_score_df

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters of DEEP extraction')
    parser.add_argument('--outcome', '-o', type=str, required=True, help='Name of outcome')
    parser.add_argument('--filename', '-p', type=str, required=True, help='Path of csv data')
    parser.add_argument('--method', '-m', type=str, default='xgb', help='Tree method used in DEEP, default = XGBoost')
    parser.add_argument('--binary_outcome', '-b', type=str2bool, default='True', help='True if the outcome is labeled in binary')
    parser.add_argument('--result_dir', '-r', type=str, default='./result_', help='True if the outcome is labeled in binary')


    args = parser.parse_args()
    outcome = args.outcome
    analyze_method = args.method
    if 'xgb' in analyze_method:
        xgb_predict = True
    else:
        xgb_predict = False
    plot_predir = '{}{}'.format(args.result_dir, analyze_method)
    if not os.path.exists(plot_predir):
        os.mkdir(plot_predir)

    pvalue_df_list = []
    pred_score_df_list = []

    binary_out = args.binary_outcome

    plotDir = plot_predir
    pvalue_df = pd.DataFrame(columns=['profile', 'outcome', 'p_val', 'relation',
                                      'coef', 'coef_95CI_lower', 'coef_95CI_upper', 'freq', 'pos_count', 'neg_count',
                                      'binary_outcome',
                                      # 'max_count',
                                      'interaction_p_vals', 'interactions_combs', 'identical_profiles_pollutants'
                                      ])
    pred_score_df = pd.DataFrame(columns=['outcome', 'binary_outcome', 'mode of min_samples_of_leaf', 'year from',
                                          'mean (std) r2 score from random predictors',
                                          'mean (std) r2 score',
                                          'mean (std) f score (minority) from random predictors',
                                          'mean (std) f score (minority)',
                                          'mean (std) f score (majority) from random predictors',
                                          'mean (std) f score (majority)',
                                          'mean (std) AUC score from random predictors',
                                          'mean (std) AUC score',
                                          'num_patients'
                                          ])

    file = args.filename
    pvalue_df, pred_score_df = runWorkflow(input_file=file,
                                           binary_outcome=binary_out,
                                           output_folder_name=outcome,
                                           p_val_df=pvalue_df,
                                           test_score_df=pred_score_df,
                                           xgb=xgb_predict
                                           )

    pvalue_df.dropna(axis=0, how='any', inplace=True)
    pvalue_df.to_csv(os.path.join(plot_predir, 'fdr_{}_{}.csv'.format(outcome, analyze_method)), index=False, header=True)
    pred_score_df.to_csv(os.path.join(plot_predir, 'pred_score_{}_{}.csv'.format(outcome, analyze_method)), index=False, header=True)
