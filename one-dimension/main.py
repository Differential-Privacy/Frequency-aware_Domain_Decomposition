import numpy as np
import pandas as pd

from treelib import Tree
from function import realfreq
from function.constructe_tree_functions import Nodex, tree_construction, non_negative, consistency_tree_with_weight_2, \
    collect_leaves, tree_query_error_recorder_by_leaf_list, leaf_node_average_level_order, consistency_adjustment, \
    consistency_tree_with_weight, non_negative_post_process, tree_query_error


def running_func(repeat_time, domain_size, theta, tree_max_height, real_frequency, query_interval_table, epsilon, data_size, alpha, beta, dataset):
    MSEDict = {'rand': []}
    repeat = 0
    mean_error=0.0
    while repeat < repeat_time:

        np.random.shuffle(dataset)

        tree = Tree()
        tree.create_node('Root', 'root',data=Nodex(1, True, 1, np.array([0, domain_size - 1]), None, None, None, 1, None, 0, 0, 1))

        tree_construction(tree, tree_max_height, theta, dataset, epsilon, alpha, data_size, beta)

        leaf_node_average_level_order(tree,epsilon,tree_max_height,data_size)

        consistency_adjustment(tree)



        leaf_list = collect_leaves(tree)
        res_buckets=non_negative_post_process(domain_size,leaf_list)
        tree_query_error(res_buckets,real_frequency, query_interval_table, domain_size, MSEDict)

        # record errors
        MSEDict_temp = pd.DataFrame.from_dict(MSEDict, orient='columns')
        MSEDict_temp.to_csv('rand_result/MSE_-Bfive-alpha{}-beta{}-eps{}.csv'.format(alpha, beta, epsilon))
        repeat += 1


def main(alpha,beta,eps):
    repeat_time=10
    data_dimension=1
    domain_size=2**9
    tree_max_height=9

    query_file='dim1_uniform_query_2^9.txt'
    query_path='Query Set/Single Dimension/'+query_file
    query_interval_table = np.loadtxt(query_path, int)
    data_name='1dim_normal_skew00'

    data_file='BFive_produced.txt'
    data_path='Data Set/'+data_file
    dataset = np.loadtxt(data_path, np.int32)
    data_size=dataset.shape[0] 

    real_frequency = realfreq.real_frequency_generation(dataset, data_size, domain_size, data_dimension,query_interval_table)  #真实的频率

    theta=float('-inf')
    running_func(repeat_time, domain_size, theta, tree_max_height, real_frequency, query_interval_table, eps, data_size, alpha, beta, dataset)


if __name__=='__main__':
    for alpha in [0.2]:
        for beta in [1]:
            for eps in [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5]:
                main(alpha,beta,eps)