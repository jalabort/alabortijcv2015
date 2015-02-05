from __future__ import division

from alabortijcv2015.result import (AlgorithmResult, FitterResult,
                                    SerializableResult)


# Concrete Implementations of AAM Algorithm Results ---------------------------

class AAMAlgorithmResult(AlgorithmResult):

    def __init__(self, image, fitter, shape_parameters, costs=0,
                 appearance_parameters=None, gt_shape=None):
        super(AAMAlgorithmResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.shape_parameters = shape_parameters
        self._costs = costs
        self.appearance_parameters = appearance_parameters
        self._gt_shape = gt_shape

    def costs(self, normalize=False):
        costs = self._costs
        if normalize:
            costs /= self._costs[0]
        return list(costs)

    @property
    def final_cost(self):
        return self.costs[-1]

    @property
    def initial_cost(self):
        return self.costs[0]


class LinearAAMAlgorithmResult(AAMAlgorithmResult):

    def __init__(self, image, fitter, shape_parameters, cost,
                 appearance_parameters=None, gt_shape=None):
        super(LinearAAMAlgorithmResult, self).__init__(
            image, fitter, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)

    def shapes(self, as_points=False):
        if as_points:
            return [self.fitter.transform.from_vector(p).sparse_target.points
                    for p in self.shape_parameters]

        else:
            return [self.fitter.transform.from_vector(p).sparse_target
                    for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target


# Concrete Implementations of AAM Fitter Results ------------------------------

class AAMFitterResult(FitterResult):

    def costs(self):
        costs = []
        for j, alg in enumerate(self.algorithm_results):
            costs += alg.costs()

        return costs

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.

        :type: `float`
        """
        return self.algorithm_results[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        return self.algorithm_results[0].initial_cost


# Serializable AAM Fitter Result ----------------------------------------------

class SerializableAAMFitterResult(SerializableResult):

    def __init__(self, image_path, shapes, costs, n_iters, algorithm,
                 gt_shape=None):
        super(SerializableAAMFitterResult, self).__init__(
            image_path, shapes, n_iters, algorithm, gt_shape=gt_shape)

        self._costs = costs

    def costs(self):
        return self._costs
