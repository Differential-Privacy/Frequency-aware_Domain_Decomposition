import consistency_method as ConMeth

class GridCell:
    def __init__(self, dimension_num = None, level = 0, cell_index = None):
        assert dimension_num != None
        self.dimension_num = dimension_num
        self.left_interval_list = [-1 for i in range(dimension_num)] # x_1, y_1, z_1
        self.right_interval_list = [-1 for i in range(dimension_num)] # x_2, y_2, z_2
        self.cell_index = cell_index
        self.dimension_index_list = None # for overall consistency
        self.level = level
        self.next_level_grid = None
        self.real_count = 0
        self.perturbed_count = 0
        self.consistent_count = 0




class Genarate_Grid:

    def __init__(self, attribute_set = None,granularity_list=None,one_dimen_flag=None,left_interval_list = None, right_interval_list = None, args = None):
        self.args = args
        self.attribute_set = attribute_set  
        self.dimension_num = len(attribute_set)
        self.Grid_index = None
        if left_interval_list is None:
            self.left_interval_list = [0 for _ in range(self.dimension_num)]
        else:
            self.left_interval_list = left_interval_list
        if right_interval_list is None:
            self.right_interval_list = [(self.args.domain_size - 1) for _ in range(self.dimension_num)]
        else:
            self.right_interval_list = right_interval_list
        self.granularity_list=granularity_list
        self.cell_unit_length_list = []
        self.cell_list = []
        self.weighted_update_matrix = []
        self.one_dimen_flag=one_dimen_flag


    def Main(self): # construct the grid
        self.generate_grid_1_dimen()


    def generate_grid_1_dimen(self):
        if self.one_dimen_flag:
            # total_cell_num = self.args.domain_size
            total_cell_num=len(self.granularity_list[self.attribute_set[0]])
        else:
            total_cell_num=len(self.granularity_list[self.attribute_set[0]])*len(self.granularity_list[self.attribute_set[1]])
        #!!!!!!
        for i in range (total_cell_num):
            new_cell = GridCell(self.dimension_num)
            new_cell.cell_index = i
            tmp_dimension_index_list = self.get_dimension_index_list_from_cell_index(new_cell.cell_index)  #获得维度信息!!!
            new_cell.dimension_index_list = tmp_dimension_index_list
            for j in range(self.dimension_num):
                tmp_dimension_index = tmp_dimension_index_list[j]
                tmp_left_interval, tmp_right_interval = self.get_left_right_interval_from_dimension_index(j, tmp_dimension_index)
                new_cell.left_interval_list[j] = tmp_left_interval
                new_cell.right_interval_list[j] = tmp_right_interval  #粒度为1时left==right
            self.cell_list.append(new_cell)
        return

    def get_dimension_index_list_from_cell_index(self, cell_index = None):
        dimension_index_list = [0 for i in range(self.dimension_num)]
        if self.one_dimen_flag:
            dimension_index_list[0] = cell_index
        else:
            dimension_index_list[1]=cell_index//len(self.granularity_list[self.attribute_set[0]])
            dimension_index_list[0] = cell_index % len(self.granularity_list[self.attribute_set[0]])
        return dimension_index_list

    def get_left_right_interval_from_dimension_index(self, dimension = None, dimension_index=None):

        if self.one_dimen_flag:
  
            return self.granularity_list[self.attribute_set[dimension]][dimension_index]
        else:
            tmp_left_interval = self.granularity_list[self.attribute_set[dimension]][dimension_index][0]
            tmp_right_interval = self.granularity_list[self.attribute_set[dimension]][dimension_index][1]
        return tmp_left_interval, tmp_right_interval

    def get_cell_index_from_attribute_value_set(self, attribute_value_set : list = None):
        tmp_cell_index = 0
        for idx_1 in range(len(self.granularity_list[self.attribute_set[1]])):
            if attribute_value_set[1]>=self.granularity_list[self.attribute_set[1]][idx_1][0] and attribute_value_set[1]<=self.granularity_list[self.attribute_set[1]][idx_1][1]:
                break
            else:
                tmp_cell_index+=len(self.granularity_list[self.attribute_set[0]])
        for idx_0 in range(len(self.granularity_list[self.attribute_set[0]])):
            if attribute_value_set[0] >= self.granularity_list[self.attribute_set[0]][idx_0][0] and attribute_value_set[0] <= self.granularity_list[self.attribute_set[0]][idx_0][1]:
                break
            else:
                tmp_cell_index += 1

        return tmp_cell_index






    def get_consistent_grid(self):
        est_value_list = [0 for i in range(len(self.cell_list))]
        for i in range(len(self.cell_list)):
            est_value_list[i] = self.cell_list[i].perturbed_count
        consistent_value_list = ConMeth.norm_sub(est_value_list, user_num=self.args.user_num)  # 非负处理

        for i in range(len(self.cell_list)):
            self.cell_list[i].consistent_count = consistent_value_list[i]
        return


    def get_consistent_grid_iteration(self):
        est_value_list = [0 for i in range(len(self.cell_list))]
        for i in range(len(self.cell_list)):
            # here is the consistent_count for iteration
            est_value_list[i] = self.cell_list[i].consistent_count
        consistent_value_list = ConMeth.norm_sub(est_value_list, user_num= self.args.user_num)
        for i in range(len(self.cell_list)):
            self.cell_list[i].consistent_count = consistent_value_list[i]
        return











    def add_real_user_record_to_grid(self, attribute_value_set : list = None):
        tmp_cell_index = self.get_cell_index_from_attribute_value_set(attribute_value_set)
        self.cell_list[tmp_cell_index].real_count += 1

    def add_perturbed_user_record_to_grid(self, attribute_value_set : list = None):
        tmp_cell_index = self.get_cell_index_from_attribute_value_set(attribute_value_set)
        self.cell_list[tmp_cell_index].perturbed_count += 1



    def get_answer_range_query_of_cell(self, cell:GridCell = None, range_query_node_list: list = None, private_flag = 0):
        query_left_interval_list = []
        query_right_interval_list = []
        for i in range(len(range_query_node_list)):
            query_left_interval_list.append(range_query_node_list[i].left_interval)
            query_right_interval_list.append(range_query_node_list[i].right_interval)
        cell_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            tmp_cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            cell_point_counts *= (tmp_cell_length + 1)  # note here 1 must be added

        overlap_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            tmp_cell_length = cell.right_interval_list[i]-cell.left_interval_list[i]
            tmp_query_length = query_right_interval_list[i] - query_left_interval_list[i]
            tmp_min_interval = min(cell.left_interval_list[i], query_left_interval_list[i])
            tmp_max_interval = max(cell.right_interval_list[i], query_right_interval_list[i])
            tmp_overlap_length = tmp_cell_length + tmp_query_length - abs(tmp_max_interval - tmp_min_interval) + 1
            if(tmp_overlap_length <= 0):
                overlap_point_counts = 0
                break
            else:
                overlap_point_counts *= tmp_overlap_length

        tans = overlap_point_counts / cell_point_counts * cell.consistent_count
        return tans


    def get_answer_range_query_of_cell_with_weight_update_matrix(self, cell:GridCell = None, range_query_node_list: list = None):
        query_left_interval_list = []
        query_right_interval_list = []

        for i in range(len(range_query_node_list)):
            query_left_interval_list.append(range_query_node_list[i].left_interval)
            query_right_interval_list.append(range_query_node_list[i].right_interval)

        cell_point_counts = 1
        for i in range(len(cell.left_interval_list)):
            tmp_cell_length = cell.right_interval_list[i] - cell.left_interval_list[i]
            cell_point_counts *= (tmp_cell_length + 1)  

        overlap_point_counts = 1
        for i in range(len(cell.left_interval_list)):  
            tmp_cell_length = cell.right_interval_list[i]-cell.left_interval_list[i]
            tmp_query_length = query_right_interval_list[i] - query_left_interval_list[i]
            tmp_min_interval = min(cell.left_interval_list[i], query_left_interval_list[i])
            tmp_max_interval = max(cell.right_interval_list[i], query_right_interval_list[i])

            tmp_overlap_length = tmp_cell_length + tmp_query_length - abs(tmp_max_interval - tmp_min_interval) + 1
            if(tmp_overlap_length <= 0):
                overlap_point_counts = 0
                break
            else:
                overlap_point_counts *= tmp_overlap_length
        tans = 0
        if overlap_point_counts == cell_point_counts:
            tans = cell.consistent_count
        elif overlap_point_counts == 0:
            tans = 0
        else:
            assert len(cell.left_interval_list) == 2
            for i in range(cell.left_interval_list[0], cell.right_interval_list[0] + 1):
                if query_left_interval_list[0] <= i and i <= query_right_interval_list[0]:
                    for j in range(cell.left_interval_list[1], cell.right_interval_list[1] + 1):
                        if query_left_interval_list[1] <= j and j <= query_right_interval_list[1]:
                            tans += self.weighted_update_matrix[i][j]
        return tans


    def answer_range_query(self, range_query_node_list):
        assert len(self.attribute_set) == len(range_query_node_list)
        ans = 0
        for i in range(len(self.cell_list)):
            tmp_cell = self.cell_list[i]
            ans += self.get_answer_range_query_of_cell(tmp_cell, range_query_node_list)
        return ans


    def answer_range_query_with_weight_update_matrix(self, range_query_node_list):
        assert len(self.attribute_set) == len(range_query_node_list)
        ans = 0
        for i in range(len(self.cell_list)):
            tmp_cell = self.cell_list[i]
            ans += self.get_answer_range_query_of_cell_with_weight_update_matrix(tmp_cell, range_query_node_list)
        return ans

