import numpy as np
from scipy import stats


class HypothesisTestDetector(object):
    METHOD = "ks"

    def __init__(self, method, window, thr):
        assert method in ["ks", "wrs", "tt"]

        if method == "ks":
            # Two-sample Kolmogorov-Smirnov test
            m = stats.ks_2samp
        elif method == "wrs":
            # Wilcoxon rank-sum test
            m = stats.ranksums
        else:
            # Two-sample t-test
            m = stats.ttest_ind

        self.method = m
        self.alarm_list = []
        self.data = []
        self.window = window
        self.thr = thr
        self.index = 0

    def add_element(self, elem):
        self.data.append(elem)

    def detected_change(self):
        x = np.array(self.data)
        w = self.window

        if len(x) < 2 * w:
            self.index += 1
            return False

        testw = x[-w:]
        refw = x[-(w * 2):-w]

        ht = self.method(testw, refw)
        pval = ht[1]
        has_change = pval < self.thr

        if has_change:
            print('Change detected at index: ' + str(self.index))
            self.alarm_list.append(self.index)
            self.index += 1
            # self.get_change = True
            self.data = []  # list(x[-w:])
            # self.data = list(x[-w:])
            return True
        else:
            self.index += 1
            return False



class FixedWindowDetector(object):

    def __init__(self, ref_window, thr, window_size):
        self.method = stats.ks_2samp
        self.alarm_list = []
        self.data = []
        self.ref_window = ref_window
        self.window_size = window_size
        self.thr = thr
        self.index = 0
        self.p_value = 1

    def add_element(self, elem):
        self.data.append(elem)
        self.ref_window.append(elem)
        self.ref_window = self.ref_window[:self.window_size]

    def detected_change(self):
        if len(self.data) < self.window_size:
            self.index += 1
            return False

        x = np.array(self.data)
        x = x[-self.window_size:]
        w = np.array(self.ref_window)

        ht = self.method(x, w)
        p_value = ht[1]
        has_change = p_value < self.thr
        self.p_value = p_value

        if has_change:
            # print('Change detected at index: ' + str(self.index))
            self.alarm_list.append(self.index)
            self.ref_window = []
            self.data = []
            self.index += 1
            return True
        else:
            self.index += 1
            return False
