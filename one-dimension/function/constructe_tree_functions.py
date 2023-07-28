import copy
import math
from collections import deque

import numpy as np
from function import errormetric
class Nodex(object):
    def __init__(self, frequency, divide_flag, count, interval, RIGHT, l_factor, r_factor, consis_fre, h_predict, user_begin, user_end, real_fre):
        self.frequency = frequency
        self.divide_flag = divide_flag
        self.count = count
        self.interval = interval
        self.RIGHT = RIGHT
        self.l_factor = l_factor
        self.r_factor = r_factor
        self.consis_fre = consis_fre
        self.h_predict = h_predict
        self.user_begin = user_begin
        self.user_end = user_end
        self.real_fre = real_fre
        self.variance=0.0


#构建树结构
def tree_construction(tree, tree_max_height, theta, dataset, epsilon, alpha, data_size, beta):
    global fading_rate
    tree_height = 0
    while tree_height < tree_max_height: 
        layer_index = 0  
        if tree_height >= 1:
            if tree_height == 1:
                fading_rate = find_fading_rate(tree)
            tree_update(tree, tree_height, theta, layer_index, epsilon, alpha, data_size, tree_max_height,dataset, fading_rate, beta)

        else: 
            tree_update(tree, tree_height, theta, layer_index, epsilon, alpha, data_size, tree_max_height,dataset, None, beta)
        tree_height += 1
    tree_real_height = tree.depth()
    print('tree_real_height', tree_real_height)
    for node in tree.all_nodes():
        if tree.level(node.identifier) == tree_real_height: 
            node.data.divide_flag = False
    for node in tree.all_nodes():
        if not node.data.divide_flag:  
            if node.data.user_end != data_size: 
                fre_rest = frequency_aggregation(epsilon, node.data.interval, dataset, node.data.user_end, data_size) 
                node.data.frequency = (node.data.frequency * (node.data.user_end - node.data.user_begin) + fre_rest * (
                        data_size - node.data.user_end)) / (data_size - node.data.user_begin)
                node.data.user_end = data_size


#叶节点平均
def leaf_node_average_level_order(tree,eps,tree_max_height,data_size):
    e_eps=np.exp(eps)
    error_oue = math.sqrt(4 * e_eps * tree_max_height / (e_eps - 1) ** 2 / data_size)
    mju=1 #放大因数
    root=tree.get_node('root')
    queue=deque()
    queue.append(root)
    for node in tree.all_nodes():
        if node.identifier=='root':
            continue
        node.data.variance=1/(node.data.user_end-node.data.user_begin)
        # node.data.v=(node.data.user_end-node.data.user_begin)

    def average_func(average_leaf_node):
        var=sum([1/(n.data.user_end-n.data.user_begin) for n in average_leaf_node])/(len(average_leaf_node)**2)
        fre=sum([n.data.frequency for n in average_leaf_node])/len(average_leaf_node)
        for node in average_leaf_node:
            node.data.variance=var
            node.data.frequency=fre
            # node.data.v=1/var
    while queue:
        node=queue.popleft()
        children=tree.children(node.identifier)
        average_leaf_node=[]
        non_leaf_node = []
        for child in children:
            if not child.data.divide_flag:
                if child.data.frequency<=mju*error_oue:
                    average_leaf_node.append(child)
                elif average_leaf_node:
                    average_func(average_leaf_node)
                    average_leaf_node=[]
            else:
                non_leaf_node.append(child)
                if average_leaf_node:
                    average_func(average_leaf_node)
                    average_leaf_node=[]
        if average_leaf_node:
            average_func(average_leaf_node)
        queue.extend(non_leaf_node)


def find_fading_rate(tree):
    fading_rate = 0
    for node in tree.leaves():
        if node.data.frequency > fading_rate:
            fading_rate = node.data.frequency
    return fading_rate


def tree_update(tree, tree_height, theta, layer_index, epsilon, alpha, data_size, tree_max_height, dataset, fading_rate, beta):
    for node in tree.leaves():
        e_eps = math.exp(epsilon)

        if not node.data.divide_flag:  
            continue
        else:  
            if tree_height == 0:  
                e_eps = math.exp(epsilon)
                n = data_size / tree_max_height  
                f = 1
                mi = math.ceil((n * f**2 * (e_eps - 1) ** 2 * alpha ** 2 / e_eps) ** (1 / 3))  
                error_oue = math.sqrt(4 * e_eps * tree_max_height / (e_eps - 1) ** 2 / data_size)  
                if mi > math.ceil(1 / (beta * error_oue)):   
                    mi = math.ceil(1 / (beta * error_oue))
                '''mi = max(2, mi) '''
                h_yuce = tree_max_height
                user_begin = 0
                user_end = user_begin + int((data_size - user_begin) / h_yuce)  
            else:
                e_eps = math.exp(epsilon)
                f = node.data.frequency

                h_yuce = height_prediction(tree_max_height, tree_height, data_size, node, fading_rate, e_eps, beta, f) 
                if tree_height == 1: 
                    update_1th_level_2(node, data_size, epsilon, dataset, h_yuce)  ##
                if h_yuce > 1:
                    user_begin = node.data.user_end
                    user_end = user_begin + int((data_size - user_begin) / (h_yuce))
                else:
                    user_begin = node.data.user_end
                    user_end = user_begin + int((data_size - user_begin) / (h_yuce + 1))
                n = user_end - user_begin
                mi = math.ceil((n * f ** 2 * (e_eps - 1) ** 2 * alpha ** 2 / e_eps) ** (1 / 3))
                error_oue = math.sqrt(4 * e_eps * h_yuce / (e_eps - 1) ** 2 / (data_size - node.data.user_end))  #噪声？？
                if mi > math.ceil(f / (beta * error_oue)):
                    mi = math.ceil(f / (beta * error_oue))

            left, right = int(node.data.interval[0]), int(node.data.interval[1])

            mi = min(mi, (right-left + 1))  
            if mi > 1: 

                length_bin = math.ceil((right - left + 1) / mi)  
                # print('mi', mi)
                for child_index in range(mi):
                    child_left = child_index * length_bin + left
                    child_right = (child_index+1) * length_bin - 1 + left
                    if child_left > right:
                        break
                    if child_right > right:
                        child_right = right
                    node_interval = [child_left, child_right]
                    node_name = str(tree_height) + str(layer_index)
                    node_frequency = frequency_aggregation(epsilon, node_interval, dataset, user_begin, user_end)
                    # node_frequency = 0
                    '''if h_yuce == 1:
                        node_divide_flag = False
                    else:
                        node_divide_flag = True'''
                    node_divide_flag = True
                    if child_right == child_left:
                        node_divide_flag = False  #后面不用再分了
                    node_count = 0
                    real_fre = count(node_interval[0], node_interval[1], dataset) / len(dataset)
                    tree.create_node(node_name, node_name, parent=node.identifier,data=Nodex(node_frequency, node_divide_flag, node_count, node_interval, None, None,None, 0, h_yuce, user_begin, user_end, real_fre))
                    layer_index += 1
            else:
                node.data.divide_flag = False  #扇出小于等于1的，就不需要再往下分了，把这个node的flag设成false
                continue


def height_prediction(tree_max_height, tree_height, data_size, node, fading_rate, e_eps, beta, f):
    h_list = []
    if tree_max_height - tree_height > 1:
        for h in range(1, tree_max_height - tree_height+1):
            # n_rest = n * (tree_max_height - tree_height)   # 临时❤
            n_rest = data_size - node.data.user_end
            if math.pow(fading_rate, 2 * h) / h >= 4 * e_eps * beta ** 2 / (
                    e_eps - 1) ** 2 / n_rest / f ** 2:  # 需不需要beta？？❤
                h_list.append(1)  # True
            else:
                h_list.append(0)  # False
        if sum(h_list) == 0:  # 全都不符合
            h_yuce = 1
        elif sum(h_list) == len(h_list):
            h_yuce = tree_max_height - tree_height
            if h_yuce > math.ceil(math.log((node.data.interval[1] - node.data.interval[0] + 1), 2)):
                h_yuce = math.ceil(math.log((node.data.interval[1] - node.data.interval[0] + 1), 2))
        else:
            h_yuce = h_list.index(0) + 1  # 继续划分..层
    else:
        h_yuce = 1
    print('h_yuce', h_yuce)
    return h_yuce


def update_1th_level_2(node, data_size, epsilon, dataset, h_yuce):
    if node.data.divide_flag: 
        h_predict = h_yuce
        if (node.data.user_end - node.data.user_begin) < int(data_size / (h_predict + 1)):
            # update frequency
            fre_rest = frequency_aggregation(epsilon, node.data.interval, dataset, node.data.user_end, int(data_size / (h_predict + 1))) #######
            node.data.frequency = (node.data.frequency * (node.data.user_end - node.data.user_begin) + fre_rest * (int(data_size / (h_predict + 1)) - node.data.user_end)) / (int(data_size / (h_predict + 1)) - node.data.user_begin)
            node.data.user_end = int(data_size / (h_predict + 1))

def frequency_aggregation(epsilon, node_interval, data_list, user_begin, user_end):
    # estimate the frequency values, and update the frequency values of the nodes
    #OUE
    p = 0.5
    q = 1.0 / (1 + math.exp(epsilon))
    number1 = count(node_interval[0], node_interval[1], data_list[user_begin: user_end])
    number = user_end - user_begin
    number0 = number - number1
    k1 = np.random.binomial(number1, p)
    k2 = np.random.binomial(number0, q)
    k = k1 + k2
    value = (k - number * q) / (p - q)
    frequency = value / number
    return frequency

def count(left, right, data):
    sum = 0
    for i in data:
        if left <= i < right + 1:
            sum += 1
    return sum

def non_negative(Domain, tree):
    for level in range(1, tree.depth() + 1):
        area = []
        non_negative_frequency = []
        parent_set = set()
        target_frequency = 0
        for node in tree.all_nodes():
            if tree.level(node.identifier) == level:
                l1 = int(node.data.interval[0])
                r1 = int(node.data.interval[1])
                area.append(r1 - l1 + 1)
                non_negative_frequency.append(node.data.frequency)
                parent = tree.parent(node.identifier)
                parent_set.add(parent)
        for p_node in parent_set:
            target_frequency += p_node.data.frequency
        while True:
            flag = 0
            zheng_area = []
            for k, frequency in enumerate(non_negative_frequency):
                if frequency < 0:
                    non_negative_frequency[k] = 0
                    flag = 1
                if frequency > 0:
                    zheng_area.append(area[k])
            if flag == 0:
                break
            values = sum(non_negative_frequency)  # 正频率值之和
            # RemainCount = len([i for i in non_negative_frequency if i > 0])  # 正频率 个数

            # delta = (target_frequency - values) / RemainCount
            for k, frequency in enumerate(non_negative_frequency):
                if frequency > 0:
                    non_negative_frequency[k] += area[k] / sum(zheng_area) * (target_frequency - values)
        k = 0
        for node in tree.all_nodes():
            if tree.level(node.identifier) == level:
                node.data.frequency = non_negative_frequency[k]
                k += 1

def non_negative_post_process(domain_size,leaf_list):
    buckets=np.zeros(domain_size)
    for leaf_node in leaf_list:
        left=leaf_node.data.interval[0]
        right=leaf_node.data.interval[1]
        buckets[left:right+1]=leaf_node.data.consis_fre/(right-left+1)

    non_negative_buckets=[x if x>0 else 0 for x in buckets]
    positive_num_A=np.sum([1 if x>0 else 0 for x in buckets])
    sum_positive_S=np.sum(non_negative_buckets)
    res_buckets= [x-(sum_positive_S-1)/positive_num_A if x>0 else 0 for x in buckets]
    return res_buckets


def tree_query_error(res_buckets, real_frequency, query_interval_table, domain_size, MSEDict):
    errList = np.zeros(len(query_interval_table))
    for i, query_interval in enumerate(query_interval_table):
        real_frequency_value = real_frequency[i]
        query_left=int(query_interval[0])
        query_right = int(query_interval[1])
        estimated_frequency_value = np.sum(res_buckets[query_left:query_right+1])
        errList[i] = real_frequency_value - estimated_frequency_value
        print('answer index {}-th query'.format(i))
        print("real_frequency_value: ", real_frequency_value)
        print("estimated_frequency_value: ", estimated_frequency_value)

    MSEDict['rand'].append(errormetric.MSE_metric(errList))


def consistency_adjustment(tree):
    for node in tree.all_nodes():
        node.data.v=1/node.data.variance if node.data.variance>0 else 0 
    for subtree_root in tree.children('root'):
        subtree=tree.subtree(subtree_root.identifier)
        for node in subtree.leaves():
            current_node=node
            leaf_z=current_node.data.v*current_node.data.frequency
            current_node.data.a=1/current_node.data.v
            current_node.data.b = 1 / current_node.data.v
            test=subtree.parent(current_node.identifier)
            while subtree.parent(current_node.identifier):
                current_parent_node=subtree.parent(current_node.identifier)
                leaf_z+=current_parent_node.data.v*current_parent_node.data.frequency
                current_node=current_parent_node
            node.data.z=leaf_z*node.data.b
        #求非叶节点的a,b,z
        # 迭代自底向上遍历
        queue = deque()
        queue.append(subtree.get_node(subtree.root))
        subtree_layer_order=[subtree.root]
        while queue:
            node = queue.popleft()
            children = subtree.children(node.identifier)
            for child in children:
                subtree_layer_order.insert(0,child.identifier)
                if not child.data.divide_flag:
                    continue
                else:
                    queue.append(child)
        for layer_order_node in subtree_layer_order:
            subtree_node = subtree.get_node(layer_order_node)
            if subtree_node.data.divide_flag:  
                sigma_children_a=0.0
                sigma_children_z = 0.0
                for child in subtree.children(layer_order_node):
                    sigma_children_a += child.data.a
                    sigma_children_z += child.data.z
                subtree_node.data.b=1/(subtree_node.data.v*sigma_children_a+1)
                subtree_node.data.a=subtree_node.data.b*sigma_children_a
                subtree_node.data.z = subtree_node.data.b * sigma_children_z

        #求调整后的f
        queue.append(subtree.get_node(subtree.root))
        while queue:
            node = queue.popleft()
            # test=node.identifier==subtree.root
            if node.identifier==subtree.root:
                node.data.consis_fre=node.data.z
                node.data.sigma_ancastor_fc_multi_v=0.0
            else:
                parent_node=subtree.parent(node.identifier)
                node.data.sigma_ancastor_fc_multi_v=parent_node.data.sigma_ancastor_fc_multi_v+parent_node.data.consis_fre*parent_node.data.v
                node.data.consis_fre = node.data.z-node.data.a*node.data.sigma_ancastor_fc_multi_v
            children = subtree.children(node.identifier)
            for child in children:
                queue.append(child)


def consistency_tree_with_weight(tree):   
    for node in tree.all_nodes():
        if not node.data.divide_flag:  # leaf
            RIGHT = node.data.frequency
            n_v = node.data.user_end - node.data.user_begin
            node_now = node
            while True:
                p_identifier = tree.parent(node_now.identifier).identifier
                n_p = tree.nodes[p_identifier].data.user_begin - tree.nodes[p_identifier].data.user_end
                RIGHT += tree.nodes[p_identifier].data.frequency * (n_p / n_v)
                if p_identifier == 'root':
                    break
                node_now = tree.nodes[p_identifier]
            node.data.RIGHT = RIGHT

    tree_nodes_reverse = reverse_tree_nodes(tree)  

    for node in tree.all_nodes():
        if not node.data.divide_flag:  # leaf

            factor_list = []
            n_v = node.data.user_end - node.data.user_begin
            node_now = node
            while True:
                p_identifier = tree.parent(node_now.identifier).identifier
                n_p = tree.nodes[p_identifier].data.user_begin - tree.nodes[p_identifier].data.user_end
                factor_list.append(n_p / n_v)
                if p_identifier == 'root':
                    break
                node_now = tree.nodes[p_identifier]
            node.data.l_factor = factor_list


    for node_id, node in tree_nodes_reverse.items():
        if node.data.divide_flag:  # non-leaf
            RIGHT = 0
            sum_child_factor = 0
            for child_node in tree.children(node.identifier):
                RIGHT += child_node.data.RIGHT
                sum_child_factor += child_node.data.l_factor[0]
            # node.data.RIGHT = RIGHT
            tree.nodes[node.identifier].data.RIGHT = RIGHT / (sum_child_factor + 1)


    for node_id, node in tree_nodes_reverse.items():
        if node.data.divide_flag:  # non-leaf
            all_factor_list = []
            sum_child_factor = 0
            new_factor_list = []
            for child_node in tree.children(node_id):
                all_factor_list.append(child_node.data.l_factor[1:])
                sum_child_factor += child_node.data.l_factor[0]
            sum_child_factor += 1

            for jjj in range(len(all_factor_list[0])):
                fac = 0
                for iii in range(len(all_factor_list)):

                    fac += all_factor_list[iii][jjj]
                new_factor_list.append(fac / sum_child_factor)
            tree.nodes[node_id].data.l_factor = new_factor_list


    for node in tree.all_nodes():
        if node.identifier == 'root':
            node.data.consis_fre = 1
        else:
            consis_ancestor = 0  # 左侧：所有父节点的一致性值*系数
            factor_list = node.data.l_factor
            node_now = node
            iii = 0
            while True:
                p_identifier = tree.parent(node_now.identifier).identifier
                consis_ancestor += tree.nodes[p_identifier].data.consis_fre * factor_list[iii]
                iii += 1
                if p_identifier == 'root':
                    break
                node_now = tree.nodes[p_identifier]

            node.data.consis_fre = node.data.RIGHT - consis_ancestor


def consistency_tree_with_weight_2(tree, domain):   
    if tree.depth() == 0:
        pass
    else:
        for node in tree.all_nodes():
            if not node.data.divide_flag:  # leaf
                RIGHT = node.data.frequency
                n_v = node.data.user_end - node.data.user_begin
                node_now = node
                while True:
                    p_identifier = tree.parent(node_now.identifier).identifier
                    if p_identifier == 'root':
                        break
                    n_p = tree.nodes[p_identifier].data.user_end - tree.nodes[p_identifier].data.user_begin
                    RIGHT += tree.nodes[p_identifier].data.frequency * (n_p / n_v)
                    '''if p_identifier == 'root':
                        break'''
                    node_now = tree.nodes[p_identifier]
                node.data.RIGHT = RIGHT

        tree_nodes_reverse = reverse_tree_nodes(tree)  

        for node in tree.all_nodes():
            if not node.data.divide_flag:  # leaf

                factor_list = []
                n_v = node.data.user_end - node.data.user_begin
                node_now = node
                while True:
                    p_identifier = tree.parent(node_now.identifier).identifier
                    if p_identifier == 'root':
                        break
                    n_p = tree.nodes[p_identifier].data.user_end - tree.nodes[p_identifier].data.user_begin
                    factor_list.append(n_p / n_v)
                    '''if p_identifier == 'root':
                        break'''
                    node_now = tree.nodes[p_identifier]
                node.data.l_factor = factor_list


        for node_id, node in tree_nodes_reverse.items():
            if node.data.divide_flag:  # non-leaf
                if node.identifier == 'root':  #### 新增
                    break
                all_factor_list = []
                sum_child_factor = 0
                new_factor_list = []
                for child_node in tree.children(node_id):
                    all_factor_list.append(child_node.data.l_factor[1:])
                    sum_child_factor += child_node.data.l_factor[0]
                sum_child_factor += 1
                for jjj in range(len(all_factor_list[0])):
                    fac = 0
                    for iii in range(len(all_factor_list)):
                        fac += all_factor_list[iii][jjj]
                    new_factor_list.append(fac / sum_child_factor)
                tree.nodes[node_id].data.l_factor = new_factor_list


        for node_id, node in tree_nodes_reverse.items():
            if node.data.divide_flag:  # non-leaf
                if node.identifier == 'root':  #### 新增
                    break
                RIGHT = 0
                sum_child_factor = 0
                for child_node in tree.children(node.identifier):
                    RIGHT += child_node.data.RIGHT
                    sum_child_factor += child_node.data.l_factor[0]
                # node.data.RIGHT = RIGHT
                tree.nodes[node.identifier].data.RIGHT = RIGHT / (sum_child_factor + 1)



        for node in tree.all_nodes():
            if node.identifier == 'root':
                continue
            if tree.parent(node.identifier).identifier == 'root':
                node.data.consis_fre = node.data.RIGHT
            else:
                consis_ancestor = 0  
                factor_list = node.data.l_factor
                node_now = node
                iii = 0
                while True:
                    p_identifier = tree.parent(node_now.identifier).identifier
                    if p_identifier == 'root':
                        break
                    consis_ancestor += tree.nodes[p_identifier].data.consis_fre * factor_list[iii]
                    iii += 1
                    '''if p_identifier == 'root':
                        break'''
                    node_now = tree.nodes[p_identifier]

                node.data.consis_fre = node.data.RIGHT - consis_ancestor


        leaf = []
        for node in tree.all_nodes():
            if not node.data.divide_flag:    # leaf
                leaf.append(node.data.consis_fre)
        delta = 1 - sum(leaf)
        for node in tree.all_nodes():
            if not node.data.divide_flag:  # leaf
                node.data.consis_fre += delta * (node.data.interval[1] - node.data.interval[0] + 1) / domain


def collect_leaves(tree):
    tree = copy.deepcopy(tree)
    leaf_list = []
    for node in tree.all_nodes():
        if not node.data.divide_flag:
            leaf_list.append(node)
    leaf_list.sort(key=lambda t: t.data.interval[0])  

    return leaf_list


def reverse_tree_nodes(tree):
    dic = tree.nodes
    keys = list(dic.keys())
    values = list(dic.values())
    keys.reverse()
    values.reverse()
    dic = dict(zip(keys, values))
    return dic



def tree_query_error_recorder_by_leaf_list(leaf_list, real_frequency, query_interval_table, domain_size, MSEDict):
    errList = np.zeros(len(query_interval_table))
    for i, query_interval in enumerate(query_interval_table):
        real_frequency_value = real_frequency[i]
        estimated_frequency_value = tree_answer_query_by_leaf_list(leaf_list, query_interval, domain_size)
        errList[i] = real_frequency_value - estimated_frequency_value
        print('answer index {}-th query'.format(i))
        print("real_frequency_value: ", real_frequency_value)
        print("estimated_frequency_value: ", estimated_frequency_value)

    MSEDict['rand'].append(errormetric.MSE_metric(errList))


def tree_answer_query_by_leaf_list(leaf_list, query_interval, domain_size):  
    estimated_frequency_value = 0
    # set 1-dim range query
    query_interval_temp = np.zeros(domain_size)
    q1_left = int(query_interval[0])
    q1_right = int(query_interval[1])
    query_interval_temp[q1_left:q1_right + 1] = 1

    for node in leaf_list:
        d1_left = int(node.data.interval[0])
        d1_right = int(node.data.interval[1])
        coeff = query_interval_temp[d1_left:d1_right + 1].sum()/(d1_right-d1_left + 1) 
        estimated_frequency_value += coeff * node.data.consis_fre
        query_interval_temp[d1_left:d1_right + 1] = 0

    return estimated_frequency_value
