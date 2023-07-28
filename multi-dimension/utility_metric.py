import math


class UtilityMetric:
    def __init__(self, args = None):
        self.args = args


    def MSE(self, real_list, est_list):
        assert len(real_list) == len(est_list)
        tans = 0
        for i in range(len(real_list)):
            tans += ((est_list[i] - real_list[i]) / self.args.user_num) ** 2
        ans = tans / len(real_list)
        return ans


if __name__ == '__main__':
    pass


