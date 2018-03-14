
import math


def getSig01(p):
    '''
        Converts a pvalue to a value where 0 is not significant and 1 is highly significant:
            0.9     => 0.0
            0.5     => 0.0
            0.099   => 0.0043
            0.01    => 0.499
            0.001   => 0.666
            0.0001  => 0.75

    :param p: p value from 0 (significant) to 1 (not significant)
    :return:
    '''
    return max(0, 1 + (1 / math.log(p, 10)))  # 1 = not sig (p=0.1), 0 = v.sig

