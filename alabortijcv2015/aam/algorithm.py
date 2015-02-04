from __future__ import division
import abc

import numpy as np
import scipy

from menpo.image import Image
from menpo.feature import gradient as fast_gradient

from .result import AAMAlgorithmResult, LinearAAMAlgorithmResult


# Abstract Interfaces for AAM Algorithms --------------------------------------

class AAMInterface(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam_algorithm):
        self.algorithm = aam_algorithm

    @abc.abstractmethod
    def dw_dp(self):
        pass

    @abc.abstractmethod
    def warp(self, image):
        pass

    @abc.abstractmethod
    def gradient(self, image):
        pass

    @abc.abstractmethod
    def steepest_descent_images(self, gradient, dw_dp):
        pass

    @abc.abstractmethod
    def partial_newton_hessian(self, gradient2, dw_dp):
        pass

    @abc.abstractmethod
    def solve(self, h, j, e, prior):
        pass


class StandardAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_step=None):
        super(StandardAAMInterface, self). __init__(aam_algorithm)

        n_true_pixels = self.algorithm.template.n_true_pixels()
        n_channels = self.algorithm.template.n_channels
        n_parameters = self.algorithm.transform.n_parameters
        sampling_mask = np.require(np.zeros(n_true_pixels), dtype=np.bool)

        if sampling_step is None:
            sampling_step = 1
        sampling_pattern = xrange(0, n_true_pixels, sampling_step)
        sampling_mask[sampling_pattern] = 1

        self.image_vec_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dw_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.gradient_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

        # self.eigenvalues = self.algorithm.transform.pdm.model.eigenvalues

    def dw_dp(self):
        dw_dp = np.rollaxis(self.algorithm.transform.d_dp(
            self.algorithm.template.mask.true_indices()), -1)
        return dw_dp[self.dw_dp_mask].reshape((dw_dp.shape[0], -1,
                                               dw_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.algorithm.template.mask,
                                  self.algorithm.transform)

    def gradient(self, image):
        g = fast_gradient(image)
        g.set_boundary_pixels()
        return g.as_vector().reshape((2, image.n_channels, -1))

    def steepest_descent_images(self, gradient, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_channels x n_pixels
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:2] + (-1,))
        # compute steepest descent images
        # gradient: n_dims x n_channels x n_pixels
        # dw_dp:    n_dims x            x n_pixels x n_params
        # sdi:               n_channels x n_pixels x n_params
        sdi = 0
        a = gradient[..., None] * dw_dp[:, None, ...]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2]))

    def partial_newton_hessian(self, gradient2, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_dims x n_channels x n_pixels
        gradient2 = gradient2[self.gradient2_mask].reshape(
            (2,) + gradient2.shape[:2] + (-1,))

        # compute partial hessian
        # gradient: n_dims x n_dims x n_channels x n_pixels
        # dw_dp:    n_dims x                     x n_pixels x n_params
        # h:                 n_dims x n_channels x n_pixels x n_params
        h1 = 0
        aux = gradient2[..., None] * dw_dp[:, None, None, ...]
        for d in aux:
            h1 += d
        # compute partial hessian
        # h:     n_dims x n_channels x n_pixels x n_params
        # dw_dp: n_dims x            x n_pixels x          x n_params
        # h:
        h2 = 0
        aux = h1[..., None] * dw_dp[..., None, :, None, :]
        for d in aux:
            h2 += d

        # reshape hessian
        # 2:  (n_channels x n_pixels) x n_params x n_params
        return h2.reshape((-1, h2.shape[3] * h2.shape[4]))

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform
        jp = t.jp()[:h.shape[0], :h.shape[0]]

        if prior:
            inv_h = np.linalg.inv(h)
            dp = inv_h.dot(j.T.dot(e))
            dp = -np.linalg.solve(t.h_prior + jp.dot(inv_h.dot(jp.T)),
                                  t.j_prior * t.as_vector() - jp.dot(dp))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))
            if jp.shape[0] is dp.shape[0]:
                dp = jp.dot(dp)
            else:
                dp = scipy.linalg.block_diag(jp, jp).dot(dp)

        return dp

    def algorithm_result(self, image, shape_parameters, cost,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class LinearAAMInterface(StandardAAMInterface):

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform

        if prior:
            dp = -np.linalg.solve(t.h_prior + h,
                                  t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_parameters, cost,
                         appearance_parameters=None, gt_shape=None):
        return LinearAAMAlgorithmResult(
            image, self.algorithm, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class PartsAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_mask=None):
        super(PartsAAMInterface, self). __init__(aam_algorithm)

        if sampling_mask is None:
            parts_shape = self.algorithm.appearance_model.parts_shape
            sampling_mask = np.require(np.ones((parts_shape)), dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.image_vec_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    def dw_dp(self):
        return np.rollaxis(self.algorithm.transform.d_dp(None), -1)

    def warp(self, image):
        return Image(image.extract_patches(
            self.algorithm.transform.target,
            patch_size=self.algorithm.appearance_model.parts_shape,
            as_single_array=True))

    def gradient(self, image):
        pixels = image.pixels
        parts_shape = self.algorithm.appearance_model.parts_shape
        g = fast_gradient(pixels.reshape((-1,) + parts_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return g.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, gradient, dw_dp):
        # reshape gradient
        # gradient: dims x parts x off x ch x (h x w)
        gradient = gradient[self.gradient_mask].reshape(
            gradient.shape[:-2] + (-1,))
        # compute steepest descent images
        # gradient: dims x parts x off x ch x (h x w)
        # ds_dp:    dims x parts x                             x params
        # sdi:             parts x off x ch x (h x w) x params
        sdi = 0
        a = gradient[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def partial_newton_hessian(self, gradient2, dw_dp):
        # reshape gradient
        # gradient: dims x dims x parts x off x ch x (h x w)
        gradient2 = gradient2[self.gradient2_mask].reshape(
            gradient2.shape[:-2] + (-1,))

        # compute partial hessian
        # gradient: dims x dims x parts x off x ch x (h x w)
        # dw_dp:    dims x      x parts x                    x params
        # h:               dims x parts x off x ch x (h x w) x params
        h1 = 0
        aux = gradient2[..., None] * dw_dp[:, None, :, None, None, None, ...]
        for d in aux:
            h1 += d
        # compute partial hessian
        # h:     dims x parts x off x ch x (h x w) x params
        # dw_dp: dims x parts x                             x params
        # h:
        h2 = 0
        aux = h1[..., None] * dw_dp[..., None, None, None, None, :]
        for d in aux:
            h2 += d

        # reshape hessian
        # 2:  (parts x off x ch x w x h) x params x params
        return h2.reshape((-1, h2.shape[-2] * h2.shape[-1]))

    def solve(self, h, j, e, prior):
        t = self.algorithm.transform

        if prior:
            dp = -np.linalg.solve(t.h_prior + h,
                                  t.j_prior * t.as_vector() - j.T.dot(e))
        else:
            dp = np.linalg.solve(h, j.T.dot(e))

        return dp

    def algorithm_result(self, image, shape_parameters, cost,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Abstract Interfaces for AAM Algorithms  -------------------------------------

class AAMAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):

        # set common state for all AAM algorithms
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform
        self.eps = eps

        # set interface
        self.interface = aam_interface(self, **kwargs)

        self._U = self.appearance_model.components.T
        self._pinv_U = np.linalg.pinv(
            self._U[self.interface.image_vec_mask, :]).T


    @abc.abstractmethod
    def precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


class ProjectOut(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(ProjectOut, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Project Out AAM algorithms
        self._masked_U = self._U[self.interface.image_vec_mask, :]

        # pre-compute
        self._precompute()
        
    def project_out(self, j):
        return j - self._masked_U.dot(self._pinv_U.T.dot(j))


class Simultaneous2(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Simultaneous, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set auxiliary template
        self.template2 = self.template.copy()

        # set common state for all Fast Simultaneous AAM algorithms
        self._masked_U = self._U[self.interface.image_vec_mask, :]

        # pre-compute
        self._precompute()

    def project_out(self, j):
        return j - self._masked_U.dot(self._pinv_U.T.dot(j))


class Alternating(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Alternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self._precompute()


class Bayesian(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):

        # call super constructor
        super(Bayesian, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # set common state for all Bayesian AAM algorithms
        self._U = self._U[self.interface.image_vec_mask, :]
        sigma2 = self.appearance_model.noise_variance()
        self._inv_sigma2 = 1 / sigma2
        self._inv_D = 1 / (self.appearance_model.eigenvalues + sigma2)

        # pre-compute
        self._precompute()

    def project_out(self, j, l=0.5):
        l_inv_sigma2 = l * self._inv_sigma2
        A = l_inv_sigma2 - (1 - l) * self._inv_D
        return 2 * (l_inv_sigma2 * j -
                    self._U.dot(A[..., None] * self._pinv_U.T.dot(j)))


# Concrete implementations of AAM Algorithms  ---------------------------------

# Project-out Compositional Algorithms ----------------------------------------

class PIC(ProjectOut):
    r"""
    Project-Out Inverse Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute inverse hessian
        self._h = self._j_po.T.dot(j)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PFC(ProjectOut):
    r"""
    Project-Out Forward Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PAC(ProjectOut):
    r"""
    Project-Out Asymmetric Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # combine image and model gradient
            nabla = a * self.interface.gradient(i) + (1-a) * self._nabla_t

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PBC(ProjectOut):
    r"""
    Project-Out Bidirectional Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(self._nabla_t,
                                                         self._dw_dp)
            j = np.hstack((j_i, j_t))
            # project out appearance model from  jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

            # save cost
            cost.append(e.T.dot(self.project_out(e)))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PIC_N(ProjectOut):
    r"""
    Project-Out Inverse Compositional Newton algorithm
    """
    def _precompute(self):

        # compute model gradient
        nabla_t = self.interface.gradient(self.template)

        # compute model second gradient
        nabla2_t = self.interface.gradient(Image(nabla_t))

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute gauss-newton hessian
        self._h_gn = self._j_po.T.dot(j)

        # compute newton hessian
        self._h_pn = self.interface.partial_newton_hessian(nabla2_t, dw_dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # project out appearance model from error
            e_po = self.project_out(e)
            # compute full newton hessian
            h = e_po.dot(self._h_pn).reshape(self._h_gn.shape) + self._h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class PFC_N(ProjectOut):
    r"""
    Project-Out Forward Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # mask model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)
            # compute image second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e)
            # compute full newton hessian
            h = e_po.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


# Simultaneous Compositional Algorithms ----------------------------------

class Simultaneous(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Simultaneous, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self.precompute()

    def precompute(self):
        r"""
        Pre-compute common state for all Simultaneous AAM algorithms
        """
        # call super method
        super(Simultaneous, self).precompute()

        # grab number of appearance parameters
        self.m = self.appearance_model.n_active_components

        # grab appearance model components
        self.A = self.appearance_model.components
        # mask them
        self.A_m = self.A[self.interface.image_vec_mask, :]
        # compute their pseudoinverse
        self.pinv_A_m = np.linalg.pinv(self.A_m)

        # grab appearance model mean
        self.a_bar = self.appearance_model.mean().as_vector()
        # masked it
        self.a_bar_m = self.a_bar[self.interface.image_vec_mask]

        # compute warp jacobian
        self.dW_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Run Simultaneous AAM algorithms
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.image_vec_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                c = self.pinv_A_m.T.dot(i_m - self.a_bar_m)
                self.a = self.a_bar + self.A.dot(c)
                a_m = self.a[self.interface.image_vec_mask]
                c_list = [c]

            # compute masked error
            e_m = a_m - i_m

            # compute masked  Jacobian
            J_m = self.compute_jacobian()
            # assemble masked simultaneous Jacobian
            J_sim_m = np.hstack((self.A_m, J_m))
            # compute masked Hessian
            H_sim_m = J_sim_m.dot(J_sim_m)

            # compute Gauss-Newton parameter update
            dq = self.interface.solve(H_sim_m, J_sim_m, e_m, map_inference)

            # update appearance parameters
            dc = dq[:self.m]
            self.a += self.A.dot(dc)
            a_m = self.a[self.interface.image_vec_mask]
            c += dc
            c_list.append(c)

            # update warp
            self.dp = dq[self.m:]
            s_k = self.transform.target
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def compute_jacobian(self):
        r"""
        Compute Jacobian
        """
        pass

    @abc.abstractmethod
    def update_warp(self):
        r"""
        Update warp
        """
        pass


class SFC(Simultaneous):
    r"""
    Simultaneous Forward Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Forward Jacobian
        """
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)


class SIC(Simultaneous):
    r"""
    Simultaneous Inverse Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute InverseJacobian
        """
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(nabla_a, self.dW_dp)


class SAC(Simultaneous):
    r"""
    Simultaneous Asymmetric Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Asymmetric Jacobian
        """
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # combine gradients
        nabla = self.a * nabla_i + (1 - self.a) * nabla_a
        # return asymmetric Jacobian
        return self.interface.steepest_descent_images(nabla, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Asymmetric Composition
        """
        dt = self.transform.from_vector(self.a * self.dp)
        dt.from_vector_inplace(dt.as_vector() + (1 - self.a) * self.dp)
        self.transform.from_vector_inplace(
            self.transform.as_vector() + dt.as_vector())

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, a=0.5):
        r"""
        Run Asymmetric Simultaneous algorithms
        """
        # set alpha value
        self.a = a
        # call super method
        return super(SAC, self).run(
            image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False)


class SBC(Simultaneous):
    r"""
    Simultaneous Bidirectional Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Bidirectional Jacobian
        """
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # compute forward Jacobian
        J_f = self.interface.steepest_descent_images(nabla_i, self.dW_dp)
        # compute inverse Jacobian
        J_i = self.interface.steepest_descent_images(nabla_a, self.dW_dp)
        # return bidirectional Jacobian
        return np.hstack((J_f, J_i))

    def update_warp(self):
        r"""
        Update warp based on Bidirectional Composition
        """
        dt = self.transform.from_vector(self.a * self.dp[:self.n])
        dt.from_vector_inplace(dt.as_vector() + self.b * self.dp[:self.n])
        self.transform.from_vector_inplace(
            self.transform.as_vector() + dt.as_vector())

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, a=1.0, b=1.0):
        r"""
        Run Bidirectional Simultaneous algorithms
        """
        # set alpha and beta values
        self.a, self.b = a, b
        # call super method
        return super(SBC, self).run(
            image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False)


class SIC_N(Simultaneous):
    r"""
    Simultaneous Inverse Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.image_vec_mask])
        self._j_U = np.zeros((self.appearance_model.n_active_components,
                              n_pixels, self.transform.n_parameters))
        for k, u in enumerate(self._U.T):
            self.template2.from_vector_inplace(u)
            nabla_u = self.interface.gradient(self.template2)
            j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
            self._j_U[k, ...] = j_u

        # compute U inverse hessian
        self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.image_vec_mask])
                dc = (self._pinv_U.T.dot(masked_e - j.dot(dp)) -
                      masked_e.dot(self._j_U).dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_i - masked_m

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model second order gradient
            nabla2_t = self.interface.gradient(Image(nabla_t))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_t, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e)
            # compute cp hessian
            h_cp = self._pinv_U.T.dot(j_po) + e_po.dot(self._j_U)

            # compute full newton hessian
            h = (e_po.dot(h_pn).reshape(h_gn.shape) + h_gn -
                 h_cp.T.dot(self._inv_h_U.dot(h_cp)))
            # compute full newton jacobian
            j = - j_po + self._pinv_U.dot(h_cp)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SFC_N(Simultaneous):
    r"""
    Simultaneous Forward Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.image_vec_mask])
        self._j_U = np.zeros((self.appearance_model.n_active_components,
                              n_pixels, self.transform.n_parameters))
        for k, u in enumerate(self._U.T):
            nabla_u = self.interface.gradient(Image(u.reshape(
                self.template.pixels.shape)))
            j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
            self._j_U[k, ...] = j_u

        # compute U inverse hessian
        self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.image_vec_mask])
                dc = (self._pinv_U.T.dot(masked_e - j.dot(dp)) -
                      masked_e.dot(self._j_U).dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_i - masked_m

            # compute model gradient
            nabla_i = self.interface.gradient(i)
            # compute model second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e)
            # compute cp hessian
            h_cp = self._pinv_U.T.dot(j_po) + e_po.dot(self._j_U)

            # compute full newton hessian
            h = (e_po.dot(h_pn).reshape(h_gn.shape) + h_gn -
                 h_cp.T.dot(self._inv_h_U.dot(h_cp)))
            # compute full newton jacobian
            j = - j_po + self._pinv_U.dot(h_cp)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Alternating Compositional Algorithms ----------------------------------------

class AIC(Alternating):
    r"""
    Alternating Inverse Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # reconstruct appearance
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AFC(Alternating):
    r"""
    Alternating Forward Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AAC(Alternating):
    r"""
    Alternating Asymmetric Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla = (a * self.interface.gradient(i) +
                     (1-a) * self.interface.gradient(self.template))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class ABC(Alternating):
    r"""
    Alternating Bidirectional Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla_i = self.interface.gradient(i)
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j = np.hstack((j_i, j_t))

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AIC_N(Alternating):
    r"""
    Alternating Inverse Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = self.template.as_vector()[self.interface.image_vec_mask] - i

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model second order gradient
            nabla2_t = self.interface.gradient(Image(nabla_t))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)

            # compute gauss-newton hessian
            h_gn = j.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_t, self._dw_dp)
            # compute full newton hessian
            h = e.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AFC_N(Alternating):
    r"""
    Alternating Forward Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_i = self.interface.gradient(i)
            # compute model second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)

            # compute gauss-newton hessian
            h_gn = j.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # compute full newton hessian
            h = e.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Modified Alternating Compositional Algorithms -------------------------------

class MAIC(Alternating):
    r"""
    Modified Alternating Inverse Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # reconstruct appearance
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class MAAC(Alternating):
    r"""
    Modified Alternating Asymmetric Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla = (a * self.interface.gradient(i) +
                     (1-a) * self.interface.gradient(self.template))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class MABC(Alternating):
    r"""
    Modified Alternating Bidirectional Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla_i = self.interface.gradient(i)
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j = np.hstack((j_i, j_t))

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# True Alternating Compositional Algorithms -----------------------------------

class TAIC(Simultaneous):
    r"""
    True Alternating Inverse Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        self._nabla_U = np.zeros((self.appearance_model.n_active_components,
                                  2, n_channels, n_true_pixels))
        for k, u in enumerate(self._U.T):
            self.template2.from_vector_inplace(u)
            nabla_u = self.interface.gradient(self.template2)
            nabla_u = nabla_u[self.interface.gradient_mask].reshape(
                nabla_u.shape[:2] + (-1,))
            self._nabla_U[k, ...] = nabla_u

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters+1):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]

                G = 0
                a = (self._nabla_U[None, ...] *
                     self._dw_dp.dot(dp)[:, None, ...])
                for ind, d in enumerate(a):
                    G += d[:, ind]
                G = G.reshape((G.shape[0], -1))

                A_hat = self._masked_U + G.T
                dc = np.linalg.pinv(A_hat).dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute and save cost
            r = t[self.interface.image_vec_mask] - masked_i
            cost.append(r.dot(r))

            if _ == max_iters:
                break

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class TAAC(Simultaneous):
    r"""
    True Alternating Asymmetric Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        self._nabla_U = np.zeros((self.appearance_model.n_active_components,
                                  2, n_channels, n_true_pixels))
        for k, u in enumerate(self._U.T):
            self.template2.from_vector_inplace(u)
            nabla_u = self.interface.gradient(self.template2)
            nabla_u = nabla_u[self.interface.gradient_mask].reshape(
                nabla_u.shape[:2] + (-1,))
            self._nabla_U[k, ...] = nabla_u

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters+1):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]

                G = 0
                a = (self._nabla_U[None, ...] *
                     self._dw_dp.dot(dp)[:, None, ...])
                for ind, d in enumerate(a):
                    G += d[:, ind]
                G = G.reshape((G.shape[0], -1))

                A_hat = self._masked_U + G.T
                dc = np.linalg.pinv(A_hat).dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute and save cost
            r = t[self.interface.image_vec_mask] - masked_i
            cost.append(r.dot(r))

            if _ == max_iters:
                break

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class TABC(Simultaneous):
    r"""
    Modified Alternating Bidirectional Compositional Gauss-Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        self._nabla_U = np.zeros((self.appearance_model.n_active_components,
                                  2, n_channels, n_true_pixels))
        for k, u in enumerate(self._U.T):
            self.template2.from_vector_inplace(u)
            nabla_u = self.interface.gradient(self.template2)
            nabla_u = nabla_u[self.interface.gradient_mask].reshape(
                nabla_u.shape[:2] + (-1,))
            self._nabla_U[k, ...] = nabla_u

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters+1):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]

                G = 0
                a = (self._nabla_U[None, ...] *
                     self._dw_dp.dot(dp)[:, None, ...])
                for ind, d in enumerate(a):
                    G += d[:, ind]
                G = G.reshape((G.shape[0], -1))

                A_hat = self._masked_U + G.T
                dc = np.linalg.pinv(A_hat).dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute and save cost
            r = t[self.interface.image_vec_mask] - masked_i
            cost.append(r.dot(r))

            if _ == max_iters:
                break

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)





























































# Simultaneous Compositional Algorithms ----------------------------------

class SIC2(Simultaneous):
    r"""
    Fast Simultaneous Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, fixed_p=None):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters+1):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                if fixed_p:
                    dp = dp[fixed_p]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute and save cost
            r = t[self.interface.image_vec_mask] - masked_i
            r2 = self.appearance_model.reconstruct(i).as_vector() - \
            i.as_vector()
            cost.append(r.dot(r))

            if _ == max_iters:
                break

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            if fixed_p:
                j = j[:, fixed_p]
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)
            if fixed_p:
                dp_ = np.zeros_like(self.transform.as_vector())
                dp_[fixed_p] = dp
                dp = dp_

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SICN(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.image_vec_mask])
        self._j_U = np.zeros((self.appearance_model.n_active_components,
                              n_pixels, self.transform.n_parameters))
        for k, u in enumerate(self._U.T):
            self.template2.from_vector_inplace(u)
            nabla_u = self.interface.gradient(self.template2)
            j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
            self._j_U[k, ...] = j_u

        # compute U inverse hessian
        self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.image_vec_mask])
                dc = (self._pinv_U.T.dot(masked_e - j.dot(dp)) -
                      masked_e.dot(self._j_U).dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_i - masked_m

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model second order gradient
            nabla2_t = self.interface.gradient(Image(nabla_t))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_t, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e)
            # compute cp hessian
            h_cp = self._pinv_U.T.dot(j_po) + e_po.dot(self._j_U)

            # compute full newton hessian
            h = (e_po.dot(h_pn).reshape(h_gn.shape) + h_gn -
                 h_cp.T.dot(self._inv_h_U.dot(h_cp)))
            # compute full newton jacobian
            j = - j_po + self._pinv_U.dot(h_cp)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SFC(Simultaneous):
    r"""
    Fast Simultaneous Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # compute model gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SFCN(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.image_vec_mask])
        self._j_U = np.zeros((self.appearance_model.n_active_components,
                              n_pixels, self.transform.n_parameters))
        for k, u in enumerate(self._U.T):
            nabla_u = self.interface.gradient(Image(u.reshape(
                self.template.pixels.shape)))
            j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
            self._j_U[k, ...] = j_u

        # compute U inverse hessian
        self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.image_vec_mask])
                dc = (self._pinv_U.T.dot(masked_e - j.dot(dp)) -
                      masked_e.dot(self._j_U).dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_i - masked_m

            # compute model gradient
            nabla_i = self.interface.gradient(i)
            # compute model second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e)
            # compute cp hessian
            h_cp = self._pinv_U.T.dot(j_po) + e_po.dot(self._j_U)

            # compute full newton hessian
            h = (e_po.dot(h_pn).reshape(h_gn.shape) + h_gn -
                 h_cp.T.dot(self._inv_h_U.dot(h_cp)))
            # compute full newton jacobian
            j = - j_po + self._pinv_U.dot(h_cp)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SSC(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # combine image and model gradient
            nabla = (a * self.interface.gradient(i) +
                     (1-a) * self.interface.gradient(self.template))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class SBC2(Simultaneous):

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_t = self.template.as_vector()[
                    self.interface.image_vec_mask]
                dc = self._pinv_U.T.dot(masked_i - masked_t + j.dot(dp))
                c += dc

            # reconstruct appearance
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = masked_m - masked_i

            # combine image and model gradient
            nabla_i = self.interface.gradient(i)
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j = np.hstack((j_i, j_t))
            # project out appearance model from model jacobian
            j_po = self.project_out(j)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[n_shape_params:])
            dt.from_vector_inplace(dt.as_vector() + dp[:n_shape_params])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Alternating Compositional Algorithms ----------------------------------------

class AIC(Alternating):
    r"""
    Alternating Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # reconstruct appearance
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AICN(Alternating):
    r"""
    Alternating Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = self.template.as_vector()[self.interface.image_vec_mask] - i

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)
            # compute model second order gradient
            nabla2_t = self.interface.gradient(Image(nabla_t))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)

            # compute gauss-newton hessian
            h_gn = j.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_t, self._dw_dp)
            # compute full newton hessian
            h = e.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AFC(Alternating):
    r"""
    Alternating Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            
            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AFCN(Alternating):
    r"""
    Alternating Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # compute model gradient
            nabla_i = self.interface.gradient(i)
            # compute model second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)

            # compute gauss-newton hessian
            h_gn = j.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # compute full newton hessian
            h = e.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class ASC(Alternating):
    r"""
    Alternating Symmetric Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla = (a * self.interface.gradient(i) +
                     (1-a) * self.interface.gradient(self.template))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class ABC(Alternating):
    r"""
    Alternating Bidirectional Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute error image
            e = (self.template.as_vector()[self.interface.image_vec_mask] -
                 masked_i)

            # combine image and model gradient
            nabla_i = self.interface.gradient(i)
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j = np.hstack((j_i, j_t))

            # compute hessian
            h = j.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e))

        # return aam algorithm result
        return self.interface.algorithm_result(
            image, shape_parameters, cost,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Bayesian Compositional Algorithms -------------------------------------------

class BIC(Bayesian):
    r"""
    Project-Out Inverse Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute inverse hessian
        self._h = self._j_po.T.dot(j)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BICN(Bayesian):
    r"""
    Project-Out Inverse Compositional Newton Algorithm
    """

    def _precompute(self):

        # compute model gradient
        nabla_t = self.interface.gradient(self.template)

        # compute model second gradient
        nabla2_t = self.interface.gradient(Image(nabla_t))

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = self.project_out(j)

        # compute gauss-newton hessian
        self._h_gn = self._j_po.T.dot(j)

        # compute newton hessian
        self._h_pn = self.interface.partial_newton_hessian(nabla2_t, dw_dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # project out appearance model from error
            e_po = self.project_out(e[..., None])[..., 0]
            # compute full newton hessian
            h = e_po.dot(self._h_pn).reshape(self._h_gn.shape) + self._h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, self._j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BFC(Bayesian):
    r"""
    Bayesian Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j, l)

            # compute hessian
            h = j_po.T.dot(j)

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BFCN(Bayesian):
    r"""
    Bayesian Forward Compositional Gauss-Newton Algorithm
    """

    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # mask model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)
            # compute image second order gradient
            nabla2_i = self.interface.gradient(Image(nabla_i))

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j, l)

            # compute gauss-newton hessian
            h_gn = j_po.T.dot(j)
            # compute partial newton hessian
            h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
            # project out appearance model from error
            e_po = self.project_out(e[..., None], l)[..., 0]
            # compute full newton hessian
            h = e_po.dot(h_pn).reshape(h_gn.shape) + h_gn

            # compute gauss-newton parameter updates
            dp = self.interface.solve(h, j, e, prior)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(e_po))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BSC(Bayesian):
    r"""
    Bayesian Symmetric Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5, a=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # combine image and model gradient
            nabla = a * self.interface.gradient(i) + (1-a) * self._nabla_t

            # compute model jacobian
            j = self.interface.steepest_descent_images(nabla, self._dw_dp)
            # project out appearance model from model jacobian
            j_po = self.project_out(j, l)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(a * dp)
            dt.from_vector_inplace(dt.as_vector() + (1-a) * dp)
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)


class BBC(Bayesian):
    r"""
    Bayesian Bidirectional Compositional ESM Algorithm
    """

    def _precompute(self):

        # compute model gradient
        self._nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, l=0.5):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        n_shape_params = self.transform.n_parameters
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute jacobian
            j_i = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
            j_t = self.interface.steepest_descent_images(self._nabla_t,
                                                         self._dw_dp)
            j = np.hstack((j_i, j_t))
            # project out appearance model from  jacobian
            j_po = self.project_out(j, l)

            # compute hessian
            h = j_po.T.dot(j)

            # compute symmetric esm parameter updates
            dp = self.interface.solve(h, j_po, e, prior)

            # update transform
            target = self.transform.target
            dt = self.transform.from_vector(dp[:n_shape_params])
            dt.from_vector_inplace(dt.as_vector() + dp[n_shape_params:])
            self.transform.from_vector_inplace(
                self.transform.as_vector() + dt.as_vector())
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            # error = np.abs(np.linalg.norm(
            #     target.points - self.transform.target.points))
            # if error < self.eps:
            #     break

            # save cost
            cost.append(e.T.dot(self.project_out(e[..., None])[..., 0]))

        # return aam algorithm result
        return self.interface.algorithm_result(image, shape_parameters, cost,
                                               gt_shape=gt_shape)
