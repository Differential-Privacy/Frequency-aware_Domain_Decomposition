import numpy as np
import pandas as pd

from treelib import Tree
from function import realfreq
from function.constructe_tree_functions import Nodex, tree_construction, non_negative, consistency_tree_with_weight_2, \
    collect_leaves, tree_query_error_recorder_by_leaf_list, leaf_node_average_level_order, consistency_adjustment, \
    consistency_tree_with_weight, non_negative_post_process, tree_query_error, level_order


def running_func(multi_dataset,attribute_num,domain_size, theta, tree_max_height, epsilon, data_size, alpha, beta,user_alpha,C_d_2,g):
        tree_list=[]
        tree_leaves_list=[]
        tree_average_leaves_list=[]
        tree_average_leaves_list_2=[]
        tree_average_leaves_tmp_old_list=[]
        res_buckets_list=[]
        for dimen_idx in range(attribute_num):
            tree = Tree()
            tree_list.append(tree)
            tree.create_node('Root', 'root',data=Nodex(1, True, 1, np.array([0, domain_size - 1]), None, None, None, 1, None, 0, 0, 1))
            dataset=[row[dimen_idx] for row in multi_dataset[(data_size//attribute_num)*dimen_idx:(data_size//attribute_num)*(dimen_idx+1)]]
            tree_construction(tree, tree_max_height, theta, dataset, epsilon, alpha, data_size//attribute_num, beta)

            tree_average_leaves_tmp_old=leaf_node_average_level_order(tree,epsilon,tree_max_height,data_size//attribute_num)
            tree_average_leaves_tmp=level_order(tree,epsilon,tree_max_height,data_size//attribute_num,user_alpha,attribute_num,C_d_2,g)
            tree_average_leaves_list.append(tree_average_leaves_tmp)
            tmp=[[i,i]for i in range(domain_size)]
            tree_average_leaves_tmp_old_list.append(tmp)


            consistency_adjustment(tree)


            leaf_list = collect_leaves(tree)
            res_buckets=non_negative_post_process(domain_size,leaf_list)


            res_buckets_list.append(res_buckets)
            tree_leaves_list.append([node.data.interval for node in leaf_list])

        return tree_list,tree_average_leaves_tmp_old_list,tree_average_leaves_list,res_buckets_list

def main(multi_dataset,attribute_num,domain_size,eps,user_alpha,C_d_2,g):
    tree_max_height=int(np.ceil(np.log2(domain_size)))
    alpha=0.2
    beta=1
    data_size=len(multi_dataset)
    theta=float('-inf')
    return running_func(multi_dataset,attribute_num,domain_size, theta, tree_max_height, eps, data_size, alpha, beta,user_alpha,C_d_2,g)
