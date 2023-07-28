import numpy as np
import utility_metric as UM
import generate_query as GenQuery
import random
import HDG
import parameter_setting as para


def setup_args(epsilon,args = None):

    args.algorithm_name = 'HDG'

    args.user_num = 0
    args.attribute_num = 5 
    args.domain_size = pow(2, 6)  #

    args.epsilon = epsilon
    args.dimension_query_volume = 0.5
    args.query_num = 200
    args.query_dimension = 2

    args.user_alpha = 0.8


def load_dataset(txt_dataset_path = None):
    user_record = []
    with open(txt_dataset_path, "r") as fr:
        i = 0
        for line in fr:
            line = line.strip()
            line = line.split()
            user_record.append(list(map(int, line)))
            i += 1
    return user_record


def sys_test(eps):
    repeat_time = 30
    args = para.generate_args() 
    setup_args(eps,args=args)  
    txt_dataset_path = "./test_dataset/Adult_dim5_domain64.txt"

    user_record = load_dataset(txt_dataset_path= txt_dataset_path) # read user data

    args.user_num = len(user_record)


    random_seed = 1
    random.seed(random_seed)
    np.random.seed(seed=random_seed)


    range_query = GenQuery.RangeQueryList(query_dimension=args.query_dimension,
                                          attribute_num=args.attribute_num,
                                          query_num=args.query_num,
                                          dimension_query_volume=args.dimension_query_volume, args=args)

    query_path = "./test_dataset/dim5_200_query_domain64.txt"
    query_interval_table = np.loadtxt(query_path, int)

    range_query.generate_range_query_list(query_interval_table)
    range_query.generate_real_answer_list(user_record)

    txt_file_path = "test_output/query_set_2.txt" 
    with open(txt_file_path, "w") as txt_fr_out:
        range_query.print_range_query_list(txt_fr_out)
    real_fre=np.zeros((args.attribute_num,args.domain_size))
    alpha_user_record=user_record[:int(args.user_alpha*args.user_num)]
    for one_record in alpha_user_record:
        for idx,val in enumerate(one_record):
            real_fre[idx][val]+=1
    real_fre/=args.user_num*args.user_alpha

    MSE_list = []
    for rep in range(repeat_time):
        #打乱用户用的
        random_seed = rep
        random.seed(random_seed)
        np.random.seed(seed=random_seed)
        np.random.shuffle(user_record)

        aa = HDG.AG_Uniform_Grid_1_2_way_optimal_2(args=args)
  
        aa.generate_attribute_group() 
        dyna_noise_data=aa.construct_Grid_set(user_record, alpha=args.user_alpha)

        aa.get_LDP_Grid_set_divide_user_2(user_record, alpha=args.user_alpha,dyna_noise_data=dyna_noise_data)


        aa.get_consistent_Grid_set_2()

        aa.get_weight_update_for_2_way_group()

        aa.answer_range_query_list(range_query.range_query_list)  


        bb = UM.UtilityMetric(args=args)

        MSE = bb.MSE(range_query.real_answer_list, aa.weighted_update_answer_list)

        MSE_list.append(MSE)

    print(np.mean(MSE_list))


if __name__ == '__main__':
    eps_list=[0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5]


    for i in eps_list:

        sys_test(i)
