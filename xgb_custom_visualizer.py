from os.path import exists, abspath, isdir
from os import mkdir
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.externals.six import StringIO
import pydotplus

def check_dir(target_dir):
    if not exists(target_dir):
        mkdir(target_dir)

def extract_split_ft_threshold_next_node(val, xgb_ft_real_dict=None, yes='yes', no='no'):
    if 'leaf' in val:
        return None
    else:
        node_rule = val.split('[')[1].split(']')[0]
        xgb_ft = node_rule.split('<')[0]
        if xgb_ft_real_dict is None:
            node_rule_replaced = node_rule
        else:
            node_rule_replaced = node_rule.replace(xgb_ft, xgb_ft_real_dict[xgb_ft])
        node_splited_by_smaller_sign = node_rule_replaced.split('<')
        node_feat = node_splited_by_smaller_sign[0]
        yes_node = val.split('yes=')[1].split(',')[0]
        no_node = val.split('no=')[1].split(',')[0]
        return {'yes': (node_feat+'<', int(yes_node)),
                'no': (node_feat+'>=', int(no_node)),
                'thres': float(node_splited_by_smaller_sign[-1]),
                'feature': node_feat,
                'node_rule':node_rule}

def visualize(clf, feature_names, labels=['0', '1'], file_name='test', plot_dir='', ext='png', save=True,
              outcome_name=''):


    if not plot_dir: plot_dir = os.path.join(os.getcwd(), 'plot')

    # ensure that labels are in string format
    labels = [str(l) for l in sorted(labels)]

    # output_path = os.path.join(plot_dir, "{}.{}".format(file_name, ext))
    # output_path = osoutcome_name)
    check_dir(outcome_name)
    output_path = os.path.join(outcome_name, "{}.{}".format(file_name, ext))

    # labels = ['0','1']
    # label_names = {'0': '-', '1': '+'}
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,  # node_ids=True,
                    special_characters=True, feature_names=feature_names)
    # ... class_names must be of string type

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    if save:
        graph.write_png(output_path)

    # Image(graph.create_png())

    return graph


def extract_booster_label_to_dict(booster_in_text, x_train, y_train, feature_idx_inv_dict):
    booster_node_list = booster_in_text.replace('\t', '').split('\n')
    label_replace_dict = {}
    b_node_dict = {}
    for bnode in booster_node_list:
        bn_split = bnode.split(':')
        if len(bn_split) == 2:
            b_node_dict[int(bn_split[0])] = bn_split[1]

    n_nodes = len(b_node_dict)
    stack = [0]
    b_node_detail = {0: {'x': x_train, 'y': y_train,
                         '#_of_patients': x_train.shape[0],
                         '#_of_positive_patients': sum(y_train),
                         '#_of_negative_patients': x_train.shape[0] - sum(y_train),
                         # '%_of_positive_patients': '{:.1f}%'.format(sum(y_train)/x_train.shape[0]*100),
                         # '%_of_negative_patients': '{:.1f}%'.format(100-sum(y_train) / x_train.shape[0] * 100),
                         }}
    not_print_list = ['x', 'y']

    while len(stack) > 0:
        node_id = stack.pop()
        split_detail = b_node_dict[node_id]
        split_dict = extract_split_ft_threshold_next_node(split_detail)
        new_node_rule = ''
        if split_dict is None:

            key_replace = split_detail
            # print(key_replace)
        else:
            yes_feat_with_sign, yes_node = split_dict['yes']
            no_feat_with_sign, no_node = split_dict['no']
            thres = split_dict['thres']
            feat = int(split_dict['feature'].replace('f', ''))
            # print(feat, split_dict['feature'])
            # feat = int(feature_idx_inv_dict[split_dict['feature']])
            # b_node_detail[node_id]['node_rule'] = split_dict['node_rule']
            key_replace = split_dict['node_rule']
            new_node_rule = split_dict['node_rule'].replace(split_dict['feature'], feature_idx_inv_dict[feat])
            x_train_in_node = b_node_detail[node_id]['x']
            y_train_in_node = b_node_detail[node_id]['y']

            yes_bool = x_train_in_node[:, feat] < thres
            no_bool = np.logical_not(yes_bool)
            yes_x, yes_y = x_train_in_node[yes_bool], y_train_in_node[yes_bool]
            no_x, no_y = x_train_in_node[no_bool], y_train_in_node[no_bool]

            b_node_detail[yes_node] = {'x': yes_x, 'y': yes_y,
                                       '#_of_patients': yes_x.shape[0],
                                       '#_of_positive_patients': sum(yes_y),
                                       '#_of_negative_patients': yes_x.shape[0] - sum(yes_y),
                                       # '%_of_positive_patients': '{:.1f}%'.format(sum(yes_y)/yes_x.shape[0]*100),
                                       # '%_of_negative_patients': '{:.1f}%'.format(100-sum(yes_y) / yes_x.shape[0] * 100)
                                       }

            b_node_detail[no_node] = {'x': no_x, 'y': no_y,
                                      '#_of_patients': no_x.shape[0],
                                      '#_of_positive_patients': sum(no_y),
                                      '#_of_negative_patients': no_x.shape[0] - sum(no_y),
                                      # '%_of_positive_patients': '{:.1f}%'.format(sum(no_y) / no_x.shape[0] * 100),
                                      # '%_of_negative_patients': '{:.1f}%'.format(100 - sum(no_y) / no_x.shape[0] * 100)
                                      }
            stack.append(yes_node)
            stack.append(no_node)
        if new_node_rule == '':
            replace_string = key_replace
        else:
            replace_string = new_node_rule
        for k, v in b_node_detail[node_id].items():
            if not k in not_print_list:
                replace_string += '\n{}:{}'.format(k, v)
        label_replace_dict[key_replace] = replace_string

    return label_replace_dict


def to_graphviz_custom(booster, training_data, outcome_name, fmap='', num_trees=0, rankdir=None,
                       yes_color=None, no_color=None,
                       condition_node_params=None, leaf_node_params=None,
                       **kwargs):
    """
    Modified from xgboost library
    """
    try:
        from graphviz import Source
        import json
    except ImportError:
        raise ImportError('You must install graphviz to plot tree')
    # if isinstance(booster, XGBModel):
    booster = booster.get_booster()

    # squash everything back into kwargs again for compatibility
    parameters_dot = 'dot'
    parameters_text = 'text'
    extra = {}
    for key, value in kwargs.items():
        extra[key] = value

    if rankdir is not None:
        kwargs['graph_attrs'] = {}
        kwargs['graph_attrs']['rankdir'] = rankdir
    for key, value in extra.items():
        if 'graph_attrs' in kwargs.keys():
            kwargs['graph_attrs'][key] = value
        else:
            kwargs['graph_attrs'] = {}
        del kwargs[key]

    if yes_color is not None or no_color is not None:
        kwargs['edge'] = {}
    if yes_color is not None:
        kwargs['edge']['yes_color'] = yes_color
    if no_color is not None:
        kwargs['edge']['no_color'] = no_color

    if condition_node_params is not None:
        kwargs['condition_node_params'] = condition_node_params
    if leaf_node_params is not None:
        kwargs['leaf_node_params'] = leaf_node_params

    if kwargs:
        parameters_dot += ':'
        parameters_dot += str(kwargs)
        parameters_text += ':'
        parameters_text += str(kwargs)
    tree_dot = booster.get_dump(
        # fmap=fmap,
        dump_format=parameters_dot)[num_trees]
    tree_text = booster.get_dump(
        # fmap=fmap,
        dump_format=parameters_text)[num_trees]
    # print(json.dump(tree_dot.__dict__))
    # print(tree_text_with_stats)
    fmap_df = pd.read_csv(fmap, sep='\t', header=None)
    # print(fmap_df.columns)
    fmap_dict = {}
    for index, row in fmap_df.iterrows():
        fmap_dict[int(row[0])] = row[1]

    label_replace_dict = extract_booster_label_to_dict(tree_text,
                                                       x_train=training_data[0],
                                                       y_train=training_data[1],
                                                       feature_idx_inv_dict=fmap_dict)
    for k, v in label_replace_dict.items():
        tree_dot = tree_dot.replace(k, v)
    tree_dot = tree_dot.replace(', missing', '')
    tree_dot = tree_dot.replace('graph [ rankdir=TB ]',
                                'graph [rankdir=TB, label="{}", labelloc=t, fontsize=30]'.format(
                                    outcome_name.split('/')[-1]))
    # tree_dot.replace("digraph {\n")
    # print(tree_dot)
    g = Source(tree_dot)

    # print(json.dumps(g.__dict__, indent=2))
    return g


def plot_tree_tiff_wrapper(booster, training_data, output_path, outcome_name, fmap='', num_trees=0, rankdir=None,
                           ax=None, **kwargs):
    """
    Modified from xgboost library
    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib import image
    except ImportError:
        raise ImportError('You must install matplotlib to plot tree')
    from io import BytesIO

    if ax is None:
        _, ax = plt.subplots(1, 1)

    g = to_graphviz_custom(booster, training_data, outcome_name=outcome_name, fmap=fmap, num_trees=num_trees,
                           rankdir=rankdir,
                           **kwargs)
    g.render(filename=output_path, cleanup=True)


def visualize_xgb(clf, feature_names, training_data, outcome_name, labels=['0', '1'], file_name='test',
                  plot_dir='', ext='png', save=True, num_trees=0,
                  tree_dir=''):
    """
    Modified from xgboost library
    """
    cNodeParams = {'shape': 'box',
                   'style': 'filled,rounded',
                   'fillcolor': '#78bceb'}
    lNodeParams = {'shape': 'box',
                   'style': 'filled',
                   'fillcolor': '#e48038'}

    fig1, ax1 = plt.subplots(1, 1, figsize=(15, 8))
    output_path = os.path.join(tree_dir, file_name)
    plot_tree_tiff_wrapper(booster=clf, fmap=feature_names, output_path=output_path, num_trees=num_trees, ax=ax1,
                           training_data=training_data, outcome_name=outcome_name,
                           **{
                               'size': str(5),
                               'condition_node_params': cNodeParams,
                               'leaf_node_params': lNodeParams
                           }
                           # conditionNodeParams=cNodeParams, leafNodeParams=lNodeParams
                           )