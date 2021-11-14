# load all libraries
from sklearn import metrics
import statsmodels.api as sm
import collections, random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
import os, fnmatch
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_curve, precision_recall_fscore_support
from xgb_custom_visualizer import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import operator
import xgboost
import random

inequality_operators = {'<': lambda x, y: x < y,
                        '<=': lambda x, y: x <= y,
                        '>': lambda x, y: x > y,
                        '>=': lambda x, y: x >= y}


plotDir = os.path.join(os.getcwd(), 'plot')

def check_dir(target_dir):
    if not exists(target_dir):
        mkdir(target_dir)



    



def get_feature_threshold_tree(estimator, counts={}, feature_names=[]):
    feature_threshold_count = {}
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    # print("The binary tree structure has %s nodes and has "
    #       "the following tree structure:"
    #       % n_nodes)
    for i in range(n_nodes):
        if not is_leaves[i]:
            # print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
            feature_name = feature_names[feature[i]]
            if not feature_name in counts:
                counts[feature_name] = []

            if not feature_name in feature_threshold_count:
                feature_threshold_count[feature_name] = 1
                counts[feature_name].append(threshold[i])
            elif feature_threshold_count[feature_name] == 1:
                # If variable appear more than once in the tree, drop it
                feature_threshold_count[feature_name] += 1
                counts[feature_name].pop()

    return counts


def parse_split(list_str, split_delimiter='\t', yes='yes', no='no'):
    root = list_str[0]
    root_rule = root.split('[')[1]
    # reach the leaf
    if len(root_rule) == 0:
        return None
    root_rule = root_rule.split(']')[0]
    yes_node = root.split('yes=')[1].split(',')[0]
    no_node = root.split('no=')[1].split(',')[0]

    yes_node_idx = 0
    no_node_idx = 0
    for str_idx, s in enumerate(list_str):
        a = 0


def count_paths_with_thres_sign_from_xgb(estimator,
                                paths={}, feature_names=[], paths_threshold = {},
                                paths_from = {},
                                labels = {},
                                merge_labels=True, to_str=False,
                                sep=' ', verbose=True, index=0, multiple_counts=False):
    feature_threshold_count = {}
    boosters = estimator.get_booster().get_dump()
    counted_path = set()
    for b_idx, b0 in enumerate(boosters):
        b0_split_n = b0.replace('\t', '').split('\n')
        xgb_ft_to_real_ft = {k: v for k, v in zip(estimator.get_booster().feature_names, feature_names)}

        b0_node_dict = {}
        for bnode in b0_split_n:
            bn_split = bnode.split(':')
            if len(bn_split) == 2:

                b0_node_dict[int(bn_split[0])] = bn_split[1]

        n_nodes = len(b0_node_dict)
        stack = [0]
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        with_contradict_sign = np.zeros(shape=n_nodes, dtype=bool)
        paths_in_this_tree = ['' for i in range(n_nodes)]
        paths_threshold_in_this_tree = {}


        while len(stack) > 0:
            node_id = stack.pop()
            split_detail = b0_node_dict[node_id]
            split_dict = extract_split_ft_threshold_next_node(split_detail, xgb_ft_to_real_ft)
            if 'leaf' in split_dict:
                is_leaves[node_id] = True
            else:

                if node_id != 0:
                    space = '\t'
                    threshold_list = paths_threshold_in_this_tree[paths_in_this_tree[node_id]] + [split_dict['thres']]
                else:
                    space = ''
                    threshold_list = [split_dict['thres']]

                base_str = paths_in_this_tree[node_id] + space
                yes_feat_with_sign, yes_node = split_dict['yes']
                no_feat_with_sign, no_node = split_dict['no']

                stack.append(yes_node)
                stack.append(no_node)

                yes_string = base_str + yes_feat_with_sign
                no_string = base_str + no_feat_with_sign

                if split_dict['feature'] in paths_in_this_tree[node_id]:
                    with_contradict_sign[yes_node] = True
                    with_contradict_sign[no_node] = True
                else:
                    with_contradict_sign[yes_node] = with_contradict_sign[node_id]
                    with_contradict_sign[no_node] = with_contradict_sign[node_id]

                paths_threshold_in_this_tree[yes_string] = threshold_list
                paths_threshold_in_this_tree[no_string] = threshold_list
                paths_in_this_tree[yes_node] = yes_string
                paths_in_this_tree[no_node] = no_string


        paths_in_this_tree = np.array(paths_in_this_tree)[is_leaves * np.logical_not(with_contradict_sign)]
        # print(paths_in_this_tree)
        for path in paths_in_this_tree:
            # print(path)
            if path != '':
                if not path in paths:
                    paths[path] = 1
                    paths_threshold[path] = [paths_threshold_in_this_tree[path]]
                    counted_path.add(path)
                    paths_from[path] = [(index, b_idx)]
                else:
                    if multiple_counts:
                        paths[path] += 1
                    else:
                        if not path in counted_path:
                            counted_path.add(path)
                            paths[path] += 1
                    paths_from[path].append((index, b_idx))
                    paths_threshold[path].append(paths_threshold_in_this_tree[path])
    # print("The binary tree structure has %s nodes and has "
    #       "the following tree structure:"
    #       % n_nodes)
    return paths, paths_threshold, paths_from


def count_paths_with_thres_sign(estimator,
                                paths={}, feature_names=[], paths_threshold = {},
                                labels = {},
                                merge_labels=True, to_str=False,
                                sep=' ', verbose=True, index=0):
    feature_threshold_count = {}
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    # print(feature)
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    with_contradict_sign = np.zeros(shape=n_nodes, dtype=bool)
    paths_in_this_tree = ['' for i in range(n_nodes)]
    paths_threshold_in_this_tree = {}



    stack = [(0, -1)]  # seed is the root node id and its parent depth
    # paths_in_this_tree = []
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
            if node_id != 0:
                space = '\t'
                threshold_list = paths_threshold_in_this_tree[paths_in_this_tree[node_id]] + [threshold[node_id]]
            else:
                space = ''
                threshold_list = [threshold[node_id]]
                # paths_threshold_in_this_tree[feature_names[feature[node_id]]] = threshold[node_id]
            # Concat string with their parent

            base_str = paths_in_this_tree[node_id] + space + feature_names[feature[node_id]]
            left_string = base_str + "<="
            right_string = base_str + ">"
            if feature_names[feature[node_id]] in paths_in_this_tree[node_id]:
                with_contradict_sign[children_left[node_id]] = True
                with_contradict_sign[children_right[node_id]] = True
            else:
                with_contradict_sign[children_left[node_id]] = with_contradict_sign[node_id]
                with_contradict_sign[children_right[node_id]] = with_contradict_sign[node_id]

            paths_threshold_in_this_tree[left_string] = threshold_list
            paths_threshold_in_this_tree[right_string] = threshold_list
            paths_in_this_tree[children_left[node_id]] = left_string
            paths_in_this_tree[children_right[node_id]] = right_string

        else:
            # print('leave with contradict sign:', with_contradict_sign[node_id])
            # print('leave path :', paths_in_this_tree[node_id])
            is_leaves[node_id] = True


    paths_in_this_tree = np.array(paths_in_this_tree)[is_leaves*np.logical_not(with_contradict_sign)]
    # print(paths_in_this_tree)
    for path in paths_in_this_tree:
        # print(path)
        if path != '':
            if not path in paths:
                paths[path] = 1
                paths_threshold[path] = [paths_threshold_in_this_tree[path]]
            else:
                paths[path] += 1
                paths_threshold[path].append(paths_threshold_in_this_tree[path])
    # print("The binary tree structure has %s nodes and has "
    #       "the following tree structure:"
    #       % n_nodes)
    return paths, paths_threshold


def load_data(input_path=None, input_file=None,
              col_target='label',
              exclude_vars=[], sep=',', confounding_vars=[], verbose=True):
    """

    Memo
    ----
    1. input file examples:
        a. multivariate imputation applied
             exposures-4yrs-merged-imputed.csv
        b. rows with 'nan' dropped
             exposures-4yrs-merged.csv
    """
    import collections
    if os.path.isdir(input_path):
        assert input_file is not None
        input_path = os.path.join(input_path, input_file)
    else:
        # if verbose: print("(load_data) input_path is a full path to the dataset: {}".format(input_path))
        pass

    assert os.path.isfile(input_path), "Invalid input path: {}".format(input_path)

    df = pd.read_csv(input_path, header=0, sep=sep)
    print(df)
    # df.drop(columns=['index'], inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    confounders_df = df[confounding_vars]
    exclude_vars = list(filter(lambda x: x in df.columns, exclude_vars))
    exclude_vars.append(col_target)
    dfx = df.drop(exclude_vars, axis=1)
    features = dfx.columns.values

    X = dfx.values
    y = df[col_target]
    print(y.shape)

    if verbose:
        counts = collections.Counter(y)
        print("... class distribution | classes: {} | sizes: {}".format(list(counts.keys()), counts))
        print("... dim(X): {} variables:".format(X.shape, len(features)))

    return (X, y, features, confounders_df, df)


def f_max(y_true, y_pred):
    y_0 = (y_true == 0).sum()
    y_1 = (y_true == 1).sum()
    if y_1 > y_0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    fscores = 2 * precision * recall / (precision + recall)
    return np.nanmax(fscores)


def f_max_thres(y_true, y_pred, thres=None, minority=True):
    y_0 = (y_true == 0).sum()
    y_1 = (y_true == 1).sum()

    if (y_1 > y_0 and minority) or ((not minority) and y_1 < y_0):
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    if thres is None:
        precision, recall, thres = precision_recall_curve(y_true, y_pred)
        fscores = 2 * precision * recall / (precision + recall)
        return np.nanmax(fscores), thres[np.where(fscores == np.nanmax(fscores))][0]
    else:
        y_pred[y_pred > thres] = 1
        y_pred[y_pred <= thres] = 0
        precision, recall, fmeasure, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        return fmeasure


f_max_score_func = make_scorer(f_max)


def features_to_txt(features, output_fn='fmap.txt'):
    output_f = open(output_fn, 'w')
    for idx, f in enumerate(features):
        output_f.write('{}\t{}\tq'.format(idx, f.replace(' ', '_')))
        if idx + 1 != len(features):
            output_f.write('\n')

    output_f.close()
    return output_fn


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # print(name)
            if fnmatch.fnmatch(name, pattern):
                # print('matched')
                result.append(name)
    return result


def classify(X, y, params={}, random_state=0, binary_outcome=True,
             analyze_method='xgb', balance_training = False, **kargs):
    assert isinstance(params, dict)

    info_gain_measure = kargs.get('criterion', 'entropy')
    xgb = kargs.get('xgb', False)

    if binary_outcome:
        if xgb is False:
            model = DecisionTreeClassifier(criterion=info_gain_measure, random_state=random_state)
        else:
            model = xgboost.XGBClassifier(random_state=random_state)
        scoring = f_max_score_func
        cv_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    else:
        if xgb is False:
            model = DecisionTreeRegressor(criterion='mse', random_state=random_state)
        else:
            model = xgboost.XGBRegressor(random_state=random_state)
        scoring = None
        cv_split = KFold(n_splits=10, shuffle=True, random_state=0)

    if xgb:
        model.fit(X, y)
        return model
    else:
        leaf = False
        pruning = True
        if leaf:
            leaf_search_space = np.append(np.array([0.05, 0.1, 0.2, 0.3, 0.4]) * y.shape[0], 1).astype(int)
        else:
            leaf_search_space = [1]
        params_list = []
        for leaf_search in leaf_search_space:
            params_dict = {'min_samples_leaf': [leaf_search]}
            model = model.set_params(**{'min_samples_leaf': leaf_search})
            model.fit(X, y)

            path = model.cost_complexity_pruning_path(X, y)
            # print(path)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            if pruning:
                params_dict['ccp_alpha'] = ccp_alphas[:-1][ccp_alphas[:-1] >= 0]
            else:
                params_dict['ccp_alpha'] = [0.0]
            params_list.append(params_dict)

        if len(ccp_alphas) <= 1:
            # print('Only 0/1 ccp_alpha value')
            return model
        else:
            final_tree = GridSearchCV(estimator=model, param_grid=params_list, cv=cv_split,
                                      scoring=scoring)
            final_tree.fit(X, y)
            return final_tree.best_estimator_


def score_collection(model, binary_outcome, X_train, y_train, X_test, y_test, scores_list):
    if binary_outcome:
        y_pred_train = model.predict_proba(X_train)[:, 1]
        _, thres_minority = f_max_thres(y_train, y_pred_train)
        _, thres_majority = f_max_thres(y_train, y_pred_train, minority=False)
        y_pred = model.predict_proba(X_test)[:, 1]
        f_minority = f_max_thres(y_test, y_pred, thres=thres_minority)
        f_majority = f_max_thres(y_test, y_pred, thres=thres_majority, minority=False)
        scores_list[0].append(f_minority)
        scores_list[1].append(f_majority)
        auc_score = metrics.roc_auc_score(y_test, y_pred)
        scores_list[2].append(auc_score)
        # score_random_shuffle = f_max(y_test, )
        print_str = "{}th Tree Fmax Score: {}"
    else:
        y_pred = model.predict(X_test)
        scoring_func = metrics.r2_score
        print_str = "{}th Tree R2 Score: {}"
        score = scoring_func(y_test, y_pred)
        scores_list[0].append(score)

    return scores_list


def analyze_path(X, y,
                 feature_set=[], n_trials=100,
                 binary_outcome=True, **kargs):
    from sklearn.model_selection import train_test_split  # Import train_test_split function
    from utils_DEEP import count_paths_with_thres_sign, count_paths_with_thres_sign_from_xgb
    import time

    #### parameters ####
    test_size = kargs.get('test_size', 0.2)
    verbose = kargs.get('verbose', False)
    plot_dir = kargs.get('plot_dir', '')
    outcome_name = kargs.get('outcome_name', '')
    xgb = kargs.get('xgb', False)
    ####################
    N, Nd = X.shape

    if len(feature_set) == 0: feature_set = ['f%s' % i for i in range(Nd)]

    msg = ''
    if verbose:
        msg += "(analyze_path) dim(X): {} | vars (n={}):\n...{}\n".format(X.shape, len(feature_set), feature_set)
        msg += "... class distribution: {}\n".format(collections.Counter(y))
    print(msg)

    paths = {}
    paths_threshold = {}
    paths_from = {}
    model_list = []

    list_min_sample_leaf = []
    list_ccp_alpha = []
    if binary_outcome:
        stratify = y
    else:
        stratify = None
    if binary_outcome:
        scores_list = [[], [], []]
        scores_rand_list = [[], [], []]
    else:
        scores_list = [[]]
        scores_rand_list = [[]]
    for i in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i,
                                                            stratify=stratify)
        model = classify(X_train, y_train, params={}, random_state=i, binary_outcome=binary_outcome, xgb=xgb)
        model_list.append(model)
        """
        Testing
        """
        scores_list = score_collection(model=model, binary_outcome=binary_outcome, X_train=X_train, y_train=y_train,
                                       y_test=y_test, X_test=X_test, scores_list=scores_list)

        for shuffle_seed in range(10):
            y_permutated_train = np.random.permutation(y_train)
            model_rand = classify(X_train, y_permutated_train, params={}, random_state=i, binary_outcome=binary_outcome,
                                  xgb=xgb)
            scores_rand_list = score_collection(model=model_rand, binary_outcome=binary_outcome, X_train=X_train,
                                                y_train=y_permutated_train,
                                                y_test=y_test, X_test=X_test, scores_list=scores_rand_list)

        if i % 10 == 0:
            print('{}th run result finished!'.format(i))

        if xgb is False:
            list_min_sample_leaf.append(model.min_samples_leaf)
            list_ccp_alpha.append(model.ccp_alpha)
            paths, paths_threshold = count_paths_with_thres_sign(estimator=model, paths=paths,
                                                                 feature_names=feature_set,
                                                                 paths_threshold=paths_threshold)
        else:
            paths, paths_threshold, paths_from = count_paths_with_thres_sign_from_xgb(estimator=model, paths=paths,
                                                                                      paths_from=paths_from, index=i,
                                                                                      feature_names=feature_set,
                                                                                      paths_threshold=paths_threshold,
                                                                                      )

    list_params = {
        'min_sample_leaf': list_min_sample_leaf,
        'ccp_alpha': list_ccp_alpha,
        # 'max_count': max_count
    }

    sorted_paths = summarize_paths(paths)

    paths_median_threshold = get_median_of_paths_threshold(paths_threshold)
    # topk = len(sorted_paths)
    topk = 10

    topk_profile_str, all_greater_path = topk_profile_with_its_threshold(sorted_paths, paths_median_threshold,
                                                                         topk=topk)
    all_greater_path_fn = os.path.join(plot_dir, '{}_all_greater_paths.csv'.format(outcome_name))
    all_greater_path_df = pd.DataFrame({'profile_name': [k for k, v in all_greater_path.items()],
                                        'count': [v for k, v in all_greater_path.items()]})
    all_greater_path_df.to_csv(all_greater_path_fn, index=False)

    return {'scores': scores_list,
            'scores_random': scores_rand_list}, list_params, \
           topk_profile_str, sorted_paths, paths_median_threshold, \
           {'paths_from': paths_from, 'model_list': model_list}


def get_median_of_paths_threshold(paths_thres):
    paths_median_threshold = {}
    for path, threshold_list in paths_thres.items():
        paths_median_threshold[path] = np.median(np.array(threshold_list), axis=0)
    return paths_median_threshold


def topk_profile_with_its_threshold(sorted_paths, paths_thres, topk, sep="\t"):
    topk_profile_with_value_str = []
    all_greater_path = {}
    for k, (path, count) in enumerate(sorted_paths):
        profile_str = ""
        if count > 10:
            for idx, pollutant in enumerate(path.split(sep)):
                # print()
                profile_str += "{}{}{:.3e}{}".format(sep, pollutant, paths_thres[path][idx], sep)
            print_str = "{}th paths ({}):{}".format(k, count, profile_str[:-1])
            topk_profile_with_value_str.append(profile_str[1:-1])
            if k < topk:
                print(print_str)
            if profile_str.count('>') > 1 and count > 1 and (not '<' in profile_str):
                # print(print_str)
                all_greater_path[profile_str] = count
    return topk_profile_with_value_str, all_greater_path
    # print("> Top {} paths (overall):\n{}\n".format(topk, sorted_paths[:topk]))


def draw_xgb_tree(test_size, split_idx, tree_dir,
                  visualize_dict, outcome_dir, fmap_fn, booster_idx, labels, X, y):
    from utils_DEEP import visualize, visualize_xgb

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=split_idx,
                                                        stratify=y)

    graph = visualize_xgb(visualize_dict['model_list'][split_idx], feature_names=fmap_fn,
                          labels=labels,
                          outcome_name=outcome_dir,
                          num_trees=booster_idx,
                          file_name="xgb{}_subtree{}".format(split_idx, booster_idx),
                          training_data=(X_train, y_train),
                          tree_dir=tree_dir)


def profile_indicator_function(path, feature_idx, path_threshold, X, sign_pair, sep='\t'):
    profile_indicator = np.ones((X.shape[0]))
    pollutant_by_order = path.split(sep)
    pollutant_ordered = path.split(sep)
    number_pollutants = len(pollutant_by_order)

    pollutants_indicators = np.ones((X.shape[0], number_pollutants))
    node_list = []
    neg_value = 0
    for n_idx, node_with_sign in enumerate(pollutant_by_order):
        node = node_with_sign.replace(sign_pair[0], '').replace(sign_pair[1], '')
        node_list.append(node_with_sign)
        if sign_pair[0] in node_with_sign:
            sign = sign_pair[1]
        else:
            sign = sign_pair[0]
        for x_idx, features in enumerate(X):
            if inequality_operators[sign](features[feature_idx[node]], path_threshold[n_idx]):
                profile_indicator[x_idx] = neg_value
                pollutants_indicators[x_idx, n_idx] = neg_value

    p_df = pd.DataFrame(pollutants_indicators, columns=node_list)
    completely_same = []
    stacked_pollutant = []
    conditions_set = []

    if number_pollutants > 1:
        pset_pollutant_dict = {}
        condition_set_pollutant_dict = {}
        for pidx in range(number_pollutants):
            append_bool = True
            if (number_pollutants > 2) and (len(pollutant_by_order) > 2):
                stacked_pollutant.append(pollutant_by_order.pop(0))
            elif number_pollutants == 2 and (len(pollutant_by_order) == 2):
                stacked_pollutant.append('')
            else:
                append_bool = False

            if append_bool:
                conditions_set.append(pollutant_by_order)
                # print(stacked_pollutant[-1])
                # print(conditions_set[-1])
                joined_str = '_and_'.join(pollutant_by_order)
                pe_array = np.ones((X.shape[0]))
                subpop_array = np.ones((X.shape[0]))
                for element in pollutant_by_order:
                    pe_array = pe_array * (p_df[element].values)

                if stacked_pollutant[-1] != '':
                    for element in stacked_pollutant:
                        subpop_array = subpop_array * (p_df[element].values)

                diff = 1
                for element in pollutant_by_order:
                    diff = min([sum(p_df[element].values - pe_array), diff])
                    # print(joined_str, element, ' diff: ', diff)
                    if diff == 0:
                        completely_same.append([joined_str, element])

                pset_pollutant_dict[joined_str] = pe_array
                condition_set_pollutant_dict['_and_'.join(stacked_pollutant)] = subpop_array

        interactions_df = pd.DataFrame(pset_pollutant_dict)
        interactions_df = 2 * interactions_df - 1

        condition_df = pd.DataFrame(condition_set_pollutant_dict)
        condition_df = 2 * condition_df - 1
    else:
        interactions_df = None
        condition_df = None

    profile_indicator = 2 * profile_indicator - 1
    p_df = 2 * p_df - 1

    # print(p_df)
    # print(interactions_df)

    return {'comb': profile_indicator,
            'pollutants_df': p_df,
            'interactions_df': interactions_df,
            'condition_df': condition_df,
            'identical_profiles': completely_same,
            'number_of_pollutant': number_pollutants,
            'pollutant_by_order': pollutant_ordered,
            'conditions_set': conditions_set
            }


def summarize_paths(paths):
    print("> 1. Frequent decision paths overall ...")
    sorted_paths = sorted(paths.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_paths


def statistical_assessment_with_confounder(sorted_paths, feature_idx_dict, paths_median_threshold,
                                           sign_pair, topk_profile_str, confounders_df,
                                           binary_outcome, y, visualize_dict, test_size, outcome_dir, fmap_fn,
                                           labels, X, possibleDirs, outcome_folder_name, file_prefix,
                                           outputDir, p_val_df, num_tree_print
                                           ):
    for idx, (profile, profile_occurrence) in enumerate(sorted_paths):
        if profile_occurrence > 10:
            profile_dict = profile_indicator_function(path=profile,
                                                      feature_idx=feature_idx_dict,
                                                      path_threshold=paths_median_threshold[profile],
                                                      X=X, sign_pair=sign_pair
                                                      )
            binary_profile = np.array(profile_dict['comb'])
            # print(profile, ' pos_count :', sum(binary_profile == 1), 'out of ', binary_profile.shape[0])
            profile_df = pd.DataFrame({topk_profile_str[idx]: binary_profile})
            regression_x_df = pd.concat([profile_df, confounders_df], axis=1)

            all_equal_drop_col = []
            for col in regression_x_df:
                unique_value = regression_x_df[col].unique()
                if len(unique_value) == 1:
                    all_equal_drop_col.append(col)

            regression_x_df_drop = regression_x_df.drop(all_equal_drop_col, axis=1)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            regression_x_df_drop[:] = scaler.fit_transform(regression_x_df_drop)
            regression_x_df_drop['intercept'] = 1.0
            if binary_outcome:
                regressor_with_confounders = sm.Logit(y, regression_x_df_drop)
            else:
                regressor_with_confounders = sm.OLS(y, regression_x_df_drop)
            result = regressor_with_confounders.fit_regularized(alpha=1e-3)
            profile_coef = result.params.values[0]
            p_val = result.pvalues.values[0]

            """Assessment of the interaction between air toxic"""

            interactions_df = profile_dict['interactions_df']
            condition_df = profile_dict['condition_df']
            interactions_pv = []
            interactions = []
            identical_to_single = profile_dict['identical_profiles']
            # if p_val < 0.05:
            if not (interactions_df is None):
                interactions = interactions_df.columns
                conditions = condition_df.columns
                for i_idx, interaction in enumerate(interactions):
                    condition = conditions[i_idx]
                    # print(condition, interaction)
                    skip_bool = False
                    for identical in identical_to_single:
                        if interaction == identical[0]:
                            skip_bool = True
                    if skip_bool == False:
                        condition_p = list(profile_dict['conditions_set'][i_idx])
                        # print(condition_p)

                        regression_pollutants_df = pd.concat([interactions_df[[interaction]],
                                                              profile_dict['pollutants_df'][condition_p],
                                                              confounders_df], axis=1)

                        regression_p_df_drop = regression_pollutants_df.drop(all_equal_drop_col, axis=1)
                        cond_bool = condition_df[condition] > 0
                        # print(cond_bool)
                        regression_p_df_drop = regression_p_df_drop.loc[cond_bool]
                        y_cond = y[cond_bool.values]

                        all_equal_drop_col_temp = []

                        for col in regression_p_df_drop:
                            if col in confounders_df.columns:
                                unique_value = regression_p_df_drop[col].unique()
                                if len(unique_value) == 1:
                                    all_equal_drop_col_temp.append(col)

                        regression_p_df_drop = regression_p_df_drop.drop(all_equal_drop_col_temp, axis=1)

                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        regression_p_df_drop[:] = scaler.fit_transform(regression_p_df_drop)

                        try:
                            X_np = np.array(regression_p_df_drop)
                            X_corr = np.corrcoef(X_np, rowvar=0)
                            # print(X_corr)
                            # print(X_corr.shape)
                            w, v = np.linalg.eig(X_corr)
                            # print('{} eigenvalues: {}'.format(interaction ,w))
                            # result = regressor_with_confounders.fit(maxiter=500, method='bfgs')
                            regression_p_df_drop['intercept'] = 1.0
                            if binary_outcome:
                                regressor_with_confounders = sm.Logit(y_cond, regression_p_df_drop)
                            else:
                                regressor_with_confounders = sm.OLS(y_cond, regression_p_df_drop)

                            result_p = regressor_with_confounders.fit_regularized(alpha=1e-3)
                            # print(result_p.summary())

                            p_val_int = result_p.pvalues.values[0]
                            interactions_pv.append(p_val_int)

                        except np.linalg.LinAlgError:
                            # p_val_int = 1
                            interactions_pv.append('repeated')

            params = result.params
            conf = result.conf_int()
            conf['Odds Ratio'] = params
            conf.columns = ['5%', '95%', 'Beta_value']
            conf_df = conf.values[0]

            if p_val < 0.05:

                p, count = (profile, profile_occurrence)
                # if count >= 5:
                path_from = visualize_dict['paths_from'][p]
                random.Random(8964).shuffle(path_from)
                p_name = '_and_'.join(p.split('\t'))

                # TODO:change to plot tree_with as much profiles (single and all_greater?)
                profile_group = ''
                profile_str = topk_profile_str[idx]
                # print(profile_str)
                if (profile_str.count('>') > 1) & (profile_str.count('<') == 0) & (profile_str.count('\t') > 0):
                    profile_group = 'all_greater'
                elif (profile_str.count('<') > 1) & (profile_str.count('>') == 0) & (
                        profile_str.count('\t') > 0):
                    profile_group = 'all_less'
                elif profile_str.count('\t') > 0:
                    profile_group = 'mixed_sign_multi_pollutants'
                elif profile_str.count('\t') == 0:
                    profile_group = 'single_pollutant'
                draw_false = False
                if (profile_group == 'all_greater') or (profile_group == 'single_pollutant'):
                    # table_count = table_count + 1
                    profile_printed = 0
                    if num_tree_print == -1:
                        tree_to_print = range(len(path_from))
                    else:
                        tree_to_print = random.Random(64).sample(range(len(path_from)), num_tree_print)
                    for path_idx, path_loc in enumerate(path_from):
                        # print('printing XGB Trees')
                        if path_idx in tree_to_print:
                            split_idx, booster_idx = path_loc
                            tree_dir = os.path.join(outputDir, p_name)
                            if not os.path.exists(tree_dir):
                                os.mkdir(tree_dir)
                            regression_x_df.to_csv(os.path.join(outputDir, '{}.csv'.format(p_name)))

                            draw_xgb_tree(test_size, split_idx, tree_dir,
                                          visualize_dict, outcome_dir, fmap_fn, booster_idx, labels, X=X, y=y)


            if (sign_pair[0] in topk_profile_str[idx] and sign_pair[1] in topk_profile_str[
                idx]) or profile_coef == 0:
                relation_dir = possibleDirs[-1]
            elif (sign_pair[0] in topk_profile_str[idx] and profile_coef < 0) or (
                    sign_pair[1] in topk_profile_str[idx] and profile_coef > 0):
                relation_dir = possibleDirs[0]
            elif (sign_pair[0] in topk_profile_str[idx] and profile_coef > 0) or (
                    sign_pair[1] in topk_profile_str[idx] and profile_coef < 0):
                relation_dir = possibleDirs[1]

            result_dir = os.path.join(relation_dir, file_prefix)

            out_path = os.path.join(result_dir, "occur_{}_{}.csv".format(profile_occurrence, topk_profile_str[idx]))

            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            opposite_profile = topk_profile_str[idx]

            opposite_profile = opposite_profile.replace(sign_pair[0], 'larger')
            opposite_profile = opposite_profile.replace(sign_pair[1], 'smaller')
            opposite_profile = opposite_profile.replace('larger', sign_pair[1])
            opposite_profile = opposite_profile.replace('smaller', sign_pair[0])

            opposite_files = find('occur_*{}_coef*.csv'.format(opposite_profile), result_dir)
            # opposite_files = [f for f in opposite_files if ' ' not in f]
            # print(topk_profile_str[idx], opposite_profile, opposite_files)
            cols = p_val_df.columns

            if (not (opposite_profile in p_val_df[[cols[0]]].values)) or (len(profile.split('\t')) > 1):
                p_val_df = p_val_df.append({cols[0]: topk_profile_str[idx],
                                            cols[1]: outcome_folder_name,
                                            cols[2]: p_val,
                                            cols[3]: relation_dir.split('/')[-1],
                                            cols[4]: profile_coef,
                                            cols[5]: conf_df[0],
                                            cols[6]: conf_df[1],
                                            cols[7]: profile_occurrence,
                                            cols[8]: sum(binary_profile == 1),
                                            cols[9]: sum(binary_profile == -1),
                                            cols[10]: binary_outcome,
                                            # cols[11]: list_params['max_count'],
                                            cols[11]: interactions_pv,
                                            cols[12]: interactions,
                                            cols[13]: identical_to_single
                                            # cols[9]: np.mean(np.array(scores)),
                                            # cols[10]: scipy.stats.mode(np.array(min_number_leaf))[0],
                                            }, ignore_index=True)

    return p_val_df
