import math

import numpy as np


class choose_granularity_beta:
    def __init__(self, args = None):
        self.args = args
        self.flag_granularity_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.alpha_1 = 0.7
        self.alpha_2 = 0.03

    def get_2_way_granularity_for_HDG_2(self, ep = None):
        n = self.args.user_num

        k = (self.args.attribute_num * (self.args.attribute_num - 1) // 2) / (1 - self.args.user_alpha)     
        x1 = 2 * self.alpha_2 * (math.exp(ep) - 1)   #2*alpha_2/(e^eps-1)
        x2 = n / (k * math.exp(ep))    #n/
        x3 = math.sqrt(x2)
        g2 = math.sqrt(x1 * x3)

        if g2 > self.args.domain_size:
            g2 = self.args.domain_size
        return g2


    def get_rounding_to_pow_2(self, gran = None):
        return gran if gran>2 else 2

        tmp_len = len(self.flag_granularity_list)


        for i in range(tmp_len - 1):
            if self.flag_granularity_list[i] <= gran and gran <= self.flag_granularity_list[i + 1]:
                dis_left = gran - self.flag_granularity_list[i]
                dis_right = self.flag_granularity_list[i + 1] - gran
                if dis_left <= dis_right:
                    if self.flag_granularity_list[i] == 1:
                        return 2
                    return self.flag_granularity_list[i]
                else:
                    return self.flag_granularity_list[i + 1]
            elif gran <= 1:
                return 2

    def get_1_way_granularity_for_HDG_2(self, ep = None):
        g1 = self.args.domain_size
        return g1






