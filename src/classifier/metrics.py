from math import pow, sqrt


class VectorMetrics:
    def __init__(self, metric_type=1):
        self._metric_type = metric_type

    def calc(self, x, y):
        if self._metric_type == 1:
            return self._cosine_sim(x, y)
        elif self._metric_type == 2:
            return self._jaccard_index(x, y)

    # noinspection PyBroadException
    @staticmethod
    def _cosine_sim(x, y):
        try:
            a = 0
            b = 0
            c = 0
            for i in range(0, len(x)):
                a += x[i] * y[i]
                b += pow(x[i], 2)
                c += pow(y[i], 2)
            return a / (sqrt(b) * sqrt(c))
        except Exception:
            # if one of the vectors is all 0 return 0
            return 0

    @staticmethod
    def _jaccard_index(x, y):
        first_set = set(x)
        second_set = set(y)
        index = 1.0
        if first_set or second_set:
            index = (float(len(first_set.intersection(second_set)))
                     / len(first_set.union(second_set)))
        return index
