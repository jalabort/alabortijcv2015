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

        # grab algorithm transform
        self.transform = self.algorithm.transform
        # grab number of shape parameters
        self.n = self.transform.n_parameters
        # grab algorithm appearance model
        self.appearance_model = self.algorithm.appearance_model
        # grab number of appearance parameters
        self.m = self.appearance_model.n_active_components

    @abc.abstractmethod
    def warp_jacobian(self):
        pass

    @abc.abstractmethod
    def warp(self, image):
        pass

    @abc.abstractmethod
    def gradient(self, image):
        pass

    @abc.abstractmethod
    def steepest_descent_images(self, nabla, dw_dp):
        pass

    @abc.abstractmethod
    def partial_newton_hessian(self, nabla2, dw_dp):
        pass

    @classmethod
    def solve_shape_map(cls, H, J, e, J_prior, p):
        if p.shape[0] is not H.shape[0]:
            # Bidirectional Compositional case
            J_prior = np.hstack((J_prior, J_prior))
            p = np.hstack((p, p))
        # compute and return MAP solution
        H += np.diag(J_prior)
        Je = J_prior * p + J.T.dot(e)
        return - np.linalg.solve(H, Je)

    @classmethod
    def solve_shape_ml(cls, H, J, e):
        # compute and return ML solution
        return -np.linalg.solve(H, J.T.dot(e))

    def solve_all_map(self, H, J, e, Ja_prior, c, Js_prior, p):
        if self.n is not H.shape[0] - self.m:
            # Bidirectional Compositional case
            Js_prior = np.hstack((Js_prior, Js_prior))
            p = np.hstack((p, p))
        # compute and return MAP solution
        J_prior = np.hstack((Ja_prior, Js_prior))
        H += np.diag(J_prior)
        Je = J_prior * np.hstack((c, p)) + J.T.dot(e)
        dq = - np.linalg.solve(H, Je)
        return dq[:self.m], dq[self.m:]

    def solve_all_ml(self, H, J, e):
        # compute ML solution
        dq = - np.linalg.solve(H, J.T.dot(e))
        return dq[:self.m], dq[self.m:]

    @abc.abstractmethod
    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        pass


class StandardAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_step=None):
        super(StandardAAMInterface, self). __init__(aam_algorithm)

        # grab algorithm shape model
        self.shape_model = self.transform.pdm.model
        # grab algorithm template
        self.template = self.algorithm.template
        # grab algorithm template mask true indices
        self.true_indices = self.template.mask.true_indices()

        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        n_parameters = self.transform.n_parameters
        sampling_mask = np.zeros(n_true_pixels, dtype=np.bool)

        if sampling_step is None:
            sampling_step = 1
        sampling_pattern = xrange(0, n_true_pixels, sampling_step)
        sampling_mask[sampling_pattern] = 1

        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dW_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.nabla_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

    def warp_jacobian(self):
        dW_dp = np.rollaxis(self.transform.d_dp(self.true_indices), -1)
        return dW_dp[self.dW_dp_mask].reshape((dW_dp.shape[0], -1,
                                               dW_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.template.mask,
                                  self.transform)

    def gradient(self, img):
        nabla = fast_gradient(img)
        nabla.set_boundary_pixels()
        return nabla.as_vector().reshape((2, img.n_channels, -1))

    def steepest_descent_images(self, nabla, dW_dp):
        # reshape gradient
        # nabla: n_dims x n_channels x n_pixels
        nabla = nabla[self.nabla_mask].reshape(nabla.shape[:2] + (-1,))
        # compute steepest descent images
        # nabla: n_dims x n_channels x n_pixels
        # warp_jacobian: n_dims x            x n_pixels x n_params
        # sdi:            n_channels x n_pixels x n_params
        sdi = 0
        a = nabla[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d
        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2])).dot(self.transform.Jp().T)

    def partial_newton_hessian(self, nabla2, dw_dp):
        # reshape gradient
        # gradient: n_dims x n_dims x n_channels x n_pixels
        nabla2 = nabla2[self.nabla2_mask].reshape(
            (2,) + nabla2.shape[:2] + (-1,))

        # compute partial hessian
        # gradient: n_dims x n_dims x n_channels x n_pixels
        # warp_jacobian:    n_dims x                     x n_pixels x n_params
        # h:                 n_dims x n_channels x n_pixels x n_params
        h1 = 0
        aux = nabla2[..., None] * dw_dp[:, None, None, ...]
        for d in aux:
            h1 += d
        # compute partial hessian
        # h:     n_dims x n_channels x n_pixels x n_params
        # warp_jacobian: n_dims x            x n_pixels x          x n_params
        # h:
        h2 = 0
        aux = h1[..., None] * dw_dp[..., None, :, None, :]
        for d in aux:
            h2 += d

        # reshape hessian
        # 2:  (n_channels x n_pixels) x n_params x n_params
        return h2.reshape((-1, h2.shape[3] * h2.shape[4]))

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class LinearAAMInterface(StandardAAMInterface):

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return LinearAAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class PartsAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling_mask=None):
        super(PartsAAMInterface, self). __init__(aam_algorithm)

        # grab algorithm shape model
        self.shape_model = self.transform.model
        # grab appearance model parts shape
        self.parts_shape = self.appearance_model.parts_shape

        if sampling_mask is None:
            sampling_mask = np.ones(self.parts_shape, dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling_mask[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    def warp_jacobian(self):
        return np.rollaxis(self.transform.d_dp(None), -1)

    def warp(self, image):
        return Image(image.extract_patches(
            self.transform.target, patch_size=self.parts_shape,
            as_single_array=True))

    def gradient(self, image):
        pixels = image.pixels
        parts_shape = self.algorithm.appearance_model.parts_shape
        g = fast_gradient(pixels.reshape((-1,) + parts_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return g.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, nabla, dw_dp):
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.gradient_mask].reshape(
            nabla.shape[:-2] + (-1,))
        # compute steepest descent images
        # nabla: dims x parts x off x ch x (h x w)
        # ds_dp:    dims x parts x                             x params
        # sdi:             parts x off x ch x (h x w) x params
        sdi = 0
        a = nabla[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def partial_newton_hessian(self, nabla2, dw_dp):
        # reshape gradient
        # gradient: dims x dims x parts x off x ch x (h x w)
        nabla2 = nabla2[self.gradient2_mask].reshape(
            nabla2.shape[:-2] + (-1,))

        # compute partial hessian
        # gradient: dims x dims x parts x off x ch x (h x w)
        # dw_dp:    dims x      x parts x                    x params
        # h:               dims x parts x off x ch x (h x w) x params
        h1 = 0
        aux = nabla2[..., None] * dw_dp[:, None, :, None, None, None, ...]
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

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# Abstract Interfaces for AAM algorithms  -------------------------------------

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
        # perform pre-computations
        self.precompute()

    def precompute(self, **kwargs):
        # grab number of shape and appearance parameters
        self.n = self.transform.n_parameters
        self.m = self.appearance_model.n_active_components

        # grab appearance model components
        self.A = self.appearance_model.components
        # mask them
        self.A_m = self.A.T[self.interface.i_mask, :]
        # compute their pseudoinverse
        self.pinv_A_m = np.linalg.pinv(self.A_m)

        # grab appearance model mean
        self.a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = self.a_bar.as_vector()[self.interface.i_mask]

        # compute warp jacobian
        self.dW_dp = self.interface.warp_jacobian()

        # compute shape model prior
        s2 = (self.appearance_model.noise_variance() /
              self.interface.shape_model.variance())
        L = self.interface.shape_model.eigenvalues
        self.s2_inv_L = np.hstack((np.ones((4,)), s2 / L))
        # compute appearance model prior
        S = self.appearance_model.eigenvalues
        self.s2_inv_S = s2 / S

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            map_inference=False):
        pass


# Abstract Interfaces for Project-out Compositional algorithms ----------------

class ProjectOut(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(ProjectOut, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

    def project_out(self, J):
        r"""
        Project-out appearance bases from a particular vector or matrix
        """
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Run Project-out AAM algorithms
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
            # vectorize it and mask it
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.a_bar_m

            # solve for increments on the shape parameters
            self.dp = self.solve(map_inference)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def solve(self, map_inference):
        pass

    @abc.abstractmethod
    def update_warp(self):
        r"""
        Update warp
        """
        pass


# Project-out Compositional Algorithms ----------------------------------------

class PFC(ProjectOut):
    r"""
    Project-out Forward Compositional Gauss-Newton algorithm
    """
    def solve(self, map_inference):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute masked forward Jacobian
        J_m = self.interface.steepest_descent_images(nabla_i, self.dW_dp)
        # project out appearance model from it
        QJ_m = self.project_out(J_m)
        # compute masked forward Hessian
        JQJ_m = QJ_m.T.dot(J_m)
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                JQJ_m, QJ_m, self.e_m,  self.s2_inv_L,
                self.transform.as_vector())
        else:
            return self.interface.solve_shape_ml(JQJ_m, QJ_m, self.e_m)

    def update_warp(self):
        r"""
        Update warp based on Forward Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class PIC(ProjectOut):
    r"""
    Project-out Inverse Compositional Gauss-Newton algorithm
    """
    def precompute(self):
        r"""
        Pre-compute PIC state
        """
        # call super method
        super(PIC, self).precompute()

        # compute appearance model mean gradient
        nabla_a = self.interface.gradient(self.a_bar)
        # compute masked inverse Jacobian
        J_m = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # project out appearance model from it
        self.QJ_m = self.project_out(J_m)
        # compute masked inverse Hessian
        self.JQJ_m = self.QJ_m.T.dot(J_m)
        # compute masked Jacobian pseudo-inverse
        self.pinv_QJ_m = np.linalg.solve(self.JQJ_m, self.QJ_m.T)

    def solve(self, map_inference):
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                self.JQJ_m, self.QJ_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return -self.pinv_QJ_m.dot(self.e_m)

    def update_warp(self):
        r"""
        Update warp based on Inverse Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class PAC(ProjectOut):
    r"""
    Project-out Asymmetric Compositional Gauss-Newton algorithm
    """
    def solve(self, map_inference):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute appearance model gradient
        nabla_a = self.interface.gradient(self.a_bar)
        # combine gradients
        nabla = self.alpha * nabla_i + (1 - self.alpha) * nabla_a
        # compute masked asymmetric Jacobian
        J_m = self.interface.steepest_descent_images(nabla, self.dW_dp)
        # project out appearance model from it
        QJ_m = self.project_out(J_m)
        # compute masked asymmetric Hessian
        JQJ_m = QJ_m.T.dot(J_m)
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                JQJ_m, QJ_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return self.interface.solve_shape_ml(JQJ_m, QJ_m, self.e_m)

    def update_warp(self):
        r"""
        Update warp based on Asymmetric Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp +
            (1 - self.alpha) * self.dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=0.5):
        r"""
        Run Asymmetric Project-out algorithm
        """
        # set alpha value
        self.alpha = alpha
        # call super method
        return super(PAC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


class PBC(ProjectOut):
    r"""
    Project-out Bidirectional Compositional Gauss-Newton algorithm
    """
    def solve(self, map_inference):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute appearance model gradient
        nabla_a = self.interface.gradient(self.a_bar)
        # compute forward Jacobian
        J_f = self.interface.steepest_descent_images(nabla_i, self.dW_dp)
        # compute inverse Jacobian
        J_i = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # assemble bidirectional Jacobian
        J_m = np.hstack((J_f, J_i))
        # project out appearance model from it
        QJ_m = self.project_out(J_m)
        # compute masked bidirectional Hessian
        JQJ_m = QJ_m.T.dot(J_m)
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                JQJ_m, QJ_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return self.interface.solve_shape_ml(JQJ_m, QJ_m, self.e_m)

    def update_warp(self):
        r"""
        Update warp based on Bidirectional Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp[:self.n] -
            self.beta * self.dp[self.n:])

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=1.0, beta=1.0):
        r"""
        Run Bidirectional Project-out algorithms
        """
        # set alpha and beta values
        self.alpha, self.beta = alpha, beta
        # call super method
        return super(PBC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


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
        dw_dp = self.interface.warp_jacobian()

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
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.i_mask]

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
        self._dw_dp = self.interface.warp_jacobian()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize cost
        cost = []
        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # mask model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.i_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

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


# Abstract Interfaces for Simultaneous Compositional algorithms ---------------

class Simultaneous(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Simultaneous, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self.precompute()

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
            i_m = self.i.as_vector()[self.interface.i_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                self.c = self.pinv_A_m.dot(i_m - self.a_bar_m)
                self.a = self.appearance_model.instance(self.c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list = [self.c]

            # compute masked error
            self.e_m = i_m - a_m

            # solve for increments on the appearance and shape parameters
            # simultaneously
            dc, self.dp = self.solve(map_inference)

            # update appearance parameters
            self.c += dc
            self.a = self.appearance_model.instance(self.c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(self.c)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def compute_jacobian(self):
        r"""
        Compute Jacobian
        """
        pass

    def solve(self, map_inference):
        # compute masked  Jacobian
        J_m = self.compute_jacobian()
        # assemble masked simultaneous Jacobian
        J_sim_m = np.hstack((-self.A_m, J_m))
        # compute masked Hessian
        H_sim_m = J_sim_m.T.dot(J_sim_m)
        # solve for increments on the appearance and shape parameters
        # simultaneously
        if map_inference:
            return self.interface.solve_all_map(
                H_sim_m, J_sim_m, self.e_m, self.s2_inv_S, self.c,
                self.s2_inv_L, self.transform.as_vector())
        else:
            return self.interface.solve_all_ml(H_sim_m, J_sim_m, self.e_m)

    @abc.abstractmethod
    def update_warp(self):
        r"""
        Update warp
        """
        pass


# Simultaneous Compositional Algorithms ----------------------------------

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

    def update_warp(self):
        r"""
        Update warp based on Forward Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class SIC(Simultaneous):
    r"""
    Simultaneous Inverse Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Inverse Jacobian
        """
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Inverse Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


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
        nabla = self.alpha * nabla_i + (1 - self.alpha) * nabla_a
        # return asymmetric Jacobian
        return self.interface.steepest_descent_images(nabla, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Asymmetric Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp +
            (1 - self.alpha) * self.dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=0.5):
        r"""
        Run Asymmetric Simultaneous algorithms
        """
        # set alpha value
        self.alpha = alpha
        # call super method
        return super(SAC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


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
        J_i = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # return bidirectional Jacobian
        return np.hstack((J_f, J_i))

    def update_warp(self):
        r"""
        Update warp based on Bidirectional Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp[:self.n] -
            self.beta * self.dp[self.n:])

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=1.0, beta=1.0):
        r"""
        Run Bidirectional Simultaneous algorithms
        """
        # set alpha and beta values
        self.alpha, self.beta = alpha, beta
        # call super method
        return super(SBC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


class SIC_N(Simultaneous):
    r"""
    Simultaneous Inverse Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.warp_jacobian()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.i_mask])
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
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.i_mask])
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
        self._dw_dp = self.interface.warp_jacobian()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.i_mask])
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
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.i_mask])
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


# Abstract Interfaces for Alternating Compositional algorithms ---------------

class Alternating(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Alternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self.precompute()

    def precompute(self, **kwargs):
        r"""
        Pre-compute common state for Alternating algorithms
        """
        # call super method
        super(Alternating, self).precompute()

        self.AA_m_map = self.A_m.T.dot(self.A_m) + np.diag(self.s2_inv_S)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Run Alternating AAM algorithms
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
            i_m = self.i.as_vector()[self.interface.i_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                c = self.pinv_A_m.dot(i_m - self.a_bar_m)
                self.a = self.appearance_model.instance(c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list = [c]
                Jdp = 0
            else:
                Jdp = J_m.dot(self.dp)

            # compute masked error
            e_m = i_m - a_m

            # solve for increment on the appearance parameters
            if map_inference:
                Ae_m_map = - self.s2_inv_S * c + self.A_m.dot(e_m + Jdp)
                dc = np.linalg.solve(self.AA_m_map, Ae_m_map)
            else:
                dc = self.pinv_A_m.dot(e_m + Jdp)

            # compute masked  Jacobian
            J_m = self.compute_jacobian()
            # compute masked Hessian
            H_m = J_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    H_m, J_m, e_m - self.A_m.T.dot(dc), self.s2_inv_L,
                    self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(H_m, J_m,
                                                        e_m - self.A_m.dot(dc))

            # update appearance parameters
            c += dc
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

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


# Alternating Compositional Algorithms ----------------------------------

class AFC(Alternating):
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

    def update_warp(self):
        r"""
        Update warp based on Forward Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class AIC(Alternating):
    r"""
    Simultaneous Inverse Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Inverse Jacobian
        """
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Inverse Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class AAC(Alternating):
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
        nabla = self.alpha * nabla_i + (1 - self.alpha) * nabla_a
        # return asymmetric Jacobian
        return self.interface.steepest_descent_images(nabla, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Asymmetric Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp +
            (1 - self.alpha) * self.dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=0.5):
        r"""
        Run Asymmetric Simultaneous algorithms
        """
        # set alpha value
        self.alpha = alpha
        # call super method
        return super(AAC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


class ABC(Alternating):
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
        J_i = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # return bidirectional Jacobian
        return np.hstack((J_f, J_i))

    def update_warp(self):
        r"""
        Update warp based on Bidirectional Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp[:self.n] -
            self.beta * self.dp[self.n:])

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=1.0, beta=1.0):
        r"""
        Run Bidirectional Simultaneous algorithms
        """
        # set alpha and beta values
        self.alpha, self.beta = alpha, beta
        # call super method
        return super(ABC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


class AIC_N(Alternating):
    r"""
    Simultaneous Inverse Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.warp_jacobian()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.i_mask])
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
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.i_mask])
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


class AFC_N(Alternating):
    r"""
    Simultaneous Forward Compositional Newton algorithm
    """
    def _precompute(self):

        # compute warp jacobian
        self._dw_dp = self.interface.warp_jacobian()

        # compute U jacobian
        n_pixels = len(self.template.as_vector()[
            self.interface.i_mask])
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
        masked_m = m[self.interface.i_mask]

        for _ in xrange(max_iters):

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.i_mask]

            if _ == 0:
                # project image onto the model bases
                c = self._pinv_U.T.dot(masked_i - masked_m)
            else:
                # compute gauss-newton appearance parameters updates
                masked_e = (masked_i -
                            self.template.as_vector()[
                                self.interface.i_mask])
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


# Abstract Interfaces for Alternating Compositional algorithms ---------------

class ModifiedAlternating(Alternating):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(ModifiedAlternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self.precompute()

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Run Alternating AAM algorithms
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        a_m = self.a_bar_m
        c_list = []
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            c = self.pinv_A_m.dot(i_m - a_m)
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # compute masked error
            e_m = i_m - a_m

            # compute masked  Jacobian
            J_m = self.compute_jacobian()
            # compute masked Hessian
            H_m = J_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    H_m, J_m, e_m, self.s2_inv_L, self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(H_m, J_m, e_m)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)


# Alternating Compositional Algorithms ----------------------------------

class MAFC(ModifiedAlternating):
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

    def update_warp(self):
        r"""
        Update warp based on Forward Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class MAIC(ModifiedAlternating):
    r"""
    Simultaneous Inverse Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Inverse Jacobian
        """
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Inverse Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class MAAC(ModifiedAlternating):
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
        nabla = self.alpha * nabla_i + (1 - self.alpha) * nabla_a
        # return asymmetric Jacobian
        return self.interface.steepest_descent_images(nabla, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Asymmetric Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp +
            (1 - self.alpha) * self.dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=0.5):
        r"""
        Run Asymmetric Simultaneous algorithms
        """
        # set alpha value
        self.alpha = alpha
        # call super method
        return super(MAAC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


class MABC(ModifiedAlternating):
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
        J_i = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # return bidirectional Jacobian
        return np.hstack((J_f, J_i))

    def update_warp(self):
        r"""
        Update warp based on Bidirectional Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp[:self.n] -
            self.beta * self.dp[self.n:])

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=1.0, beta=1.0):
        r"""
        Run Bidirectional Simultaneous algorithms
        """
        # set alpha and beta values
        self.alpha, self.beta = alpha, beta
        # call super method
        return super(MABC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


# Abstract Interfaces for Alternating Compositional algorithms ---------------

class Wiberg(AAMAlgorithm):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Wiberg, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

        # pre-compute
        self.precompute()

    def project_out(self, J):
        r"""
        Project-out appearance bases from a particular vector or matrix
        """
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Run Alternating AAM algorithms
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
            i_m = self.i.as_vector()[self.interface.i_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                c = self.pinv_A_m.dot(i_m - self.a_bar_m)
                self.a = self.appearance_model.instance(c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list = [c]
            else:
                c = self.pinv_A_m.dot(i_m - a_m + J_m.dot(self.dp))
                self.a = self.appearance_model.instance(c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list.append(c)

            # compute masked error
            e_m = i_m - self.a_bar_m

            # compute masked  Jacobian
            J_m = self.compute_jacobian()
            # project out appearance models
            QJ_m = self.project_out(J_m)
            # compute masked Hessian
            JQJ_m = QJ_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    JQJ_m, QJ_m, e_m, self.s2_inv_L,
                    self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(JQJ_m, QJ_m, e_m)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)


# Alternating Compositional Algorithms ----------------------------------

class WFC(Wiberg):
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

    def update_warp(self):
        r"""
        Update warp based on Forward Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class WIC(Wiberg):
    r"""
    Simultaneous Inverse Compositional Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        r"""
        Compute Inverse Jacobian
        """
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Inverse Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class WAC(Wiberg):
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
        nabla = self.alpha * nabla_i + (1 - self.alpha) * nabla_a
        # return asymmetric Jacobian
        return self.interface.steepest_descent_images(nabla, self.dW_dp)

    def update_warp(self):
        r"""
        Update warp based on Asymmetric Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp +
            (1 - self.alpha) * self.dp)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=0.5):
        r"""
        Run Asymmetric Simultaneous algorithms
        """
        # set alpha value
        self.alpha = alpha
        # call super method
        return super(WAC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


class WBC(Wiberg):
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
        J_i = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # return bidirectional Jacobian
        return np.hstack((J_f, J_i))

    def update_warp(self):
        r"""
        Update warp based on Bidirectional Composition
        """
        self.transform.from_vector_inplace(
            self.transform.as_vector() +
            self.alpha * self.dp[:self.n] -
            self.beta * self.dp[self.n:])

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False, alpha=1.0, beta=1.0):
        r"""
        Run Bidirectional Simultaneous algorithms
        """
        # set alpha and beta values
        self.alpha, self.beta = alpha, beta
        # call super method
        return super(WBC, self).run(
            image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
            map_inference=map_inference)


























# # Abstract Interfaces for Alternating Compositional algorithms ---------------
#
# class ModifiedAlternating(Alternating):
#
#     def __init__(self, aam_interface, appearance_model, transform,
#                  eps=10**-5, **kwargs):
#         # call super constructor
#         super(ModifiedAlternating, self).__init__(
#             aam_interface, appearance_model, transform, eps, **kwargs)
#
#         # pre-compute
#         self.precompute()
#
#     def run(self, image, initial_shape, gt_shape=None, max_iters=20,
#             map_inference=False):
#         r"""
#         Run Alternating AAM algorithms
#         """
#         # initialize transform
#         self.transform.set_target(initial_shape)
#         p_list = [self.transform.as_vector()]
#
#         # initialize iteration counter and epsilon
#         k = 0
#         eps = np.Inf
#
#         # Compositional Gauss-Newton loop
#         while k < max_iters and eps > self.eps:
#             # warp image
#             self.i = self.interface.warp(image)
#             # mask warped image
#             i_m = self.i.as_vector()[self.interface.i_mask]
#
#             if k == 0:
#                 # initialize appearance parameters by projecting masked image
#                 # onto masked appearance model
#                 c = self.pinv_A_m.dot(i_m - self.a_bar_m)
#                 self.a = self.appearance_model.instance(c)
#                 a_m = self.a.as_vector()[self.interface.i_mask]
#                 c_list = [c]
#                 Jdp = 0
#             else:
#                 Jdp = J_m.dot(self.dp)
#
#             # compute masked error
#             e_m = i_m - a_m
#
#             # solve for increment on the appearance parameters
#             if map_inference:
#                 Ae_m_map = - self.s2_inv_S * c + self.A_m.T.dot(e_m + Jdp)
#                 dc = np.linalg.solve(self.AA_m_map, Ae_m_map)
#             else:
#                 dc = self.pinv_A_m.dot(e_m + Jdp)
#
#             print c
#
#             # update appearance parameters
#             c += dc
#             self.a = self.appearance_model.instance(c)
#             a_m = self.a.as_vector()[self.interface.i_mask]
#             c_list.append(c)
#
#             # compute masked  Jacobian
#             J_m = self.compute_jacobian()
#             # compute masked Hessian
#             H_m = J_m.T.dot(J_m)
#             # solve for increments on the shape parameters
#             if map_inference:
#                 self.dp = self.interface.solve_shape_map(
#                     H_m, J_m, e_m, np.diag(self.s2_inv_L), self.s2_inv_L,
#                     self.transform.as_vector())
#             else:
#                 self.dp = self.interface.solve_shape_ml(H_m, J_m, e_m)
#
#             # update warp
#             s_k = self.transform.target.points
#             self.update_warp()
#             p_list.append(self.transform.as_vector())
#
#             # test convergence
#             eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))
#
#             # increase iteration counter
#             k += 1
#
#         # return algorithm result
#         return self.interface.algorithm_result(
#             image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)
#
#
# # Alternating Compositional Algorithms ----------------------------------
#
# class MAFC(ModifiedAlternating):
#     r"""
#     Simultaneous Forward Compositional Gauss-Newton algorithm
#     """
#     def compute_jacobian(self):
#         r"""
#         Compute Forward Jacobian
#         """
#         # compute warped image gradient
#         nabla_i = self.interface.gradient(self.i)
#         # return forward Jacobian
#         return self.interface.steepest_descent_images(nabla_i, self.dW_dp)
#
#     def update_warp(self):
#         r"""
#         Update warp based on Forward Composition
#         """
#         self.transform.from_vector_inplace(
#             self.transform.as_vector() + self.dp)
#
#
# class MAIC(ModifiedAlternating):
#     r"""
#     Simultaneous Inverse Compositional Gauss-Newton algorithm
#     """
#     def compute_jacobian(self):
#         r"""
#         Compute Inverse Jacobian
#         """
#         # compute warped appearance model gradient
#         nabla_a = self.interface.gradient(self.a)
#         # return inverse Jacobian
#         return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
#
#     def update_warp(self):
#         r"""
#         Update warp based on Inverse Composition
#         """
#         self.transform.from_vector_inplace(
#             self.transform.as_vector() - self.dp)
#
#
# class MAAC(ModifiedAlternating):
#     r"""
#     Simultaneous Asymmetric Compositional Gauss-Newton algorithm
#     """
#     def compute_jacobian(self):
#         r"""
#         Compute Asymmetric Jacobian
#         """
#         # compute warped image gradient
#         nabla_i = self.interface.gradient(self.i)
#         # compute appearance model gradient
#         nabla_a = self.interface.gradient(self.a)
#         # combine gradients
#         nabla = self.alpha * nabla_i + (1 - self.alpha) * nabla_a
#         # return asymmetric Jacobian
#         return self.interface.steepest_descent_images(nabla, self.dW_dp)
#
#     def update_warp(self):
#         r"""
#         Update warp based on Asymmetric Composition
#         """
#         self.transform.from_vector_inplace(
#             self.transform.as_vector() +
#             self.alpha * self.dp +
#             (1 - self.alpha) * self.dp)
#
#     def run(self, image, initial_shape, gt_shape=None, max_iters=20,
#             map_inference=False, alpha=0.5):
#         r"""
#         Run Asymmetric Simultaneous algorithms
#         """
#         # set alpha value
#         self.alpha = alpha
#         # call super method
#         return super(AAC, self).run(
#             image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
#             map_inference=map_inference)
#
#
# class MABC(ModifiedAlternating):
#     r"""
#     Simultaneous Bidirectional Compositional Gauss-Newton algorithm
#     """
#     def compute_jacobian(self):
#         r"""
#         Compute Bidirectional Jacobian
#         """
#         # compute warped image gradient
#         nabla_i = self.interface.gradient(self.i)
#         # compute appearance model gradient
#         nabla_a = self.interface.gradient(self.a)
#         # compute forward Jacobian
#         J_f = self.interface.steepest_descent_images(nabla_i, self.dW_dp)
#         # compute inverse Jacobian
#         J_i = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
#         # return bidirectional Jacobian
#         return np.hstack((J_f, J_i))
#
#     def update_warp(self):
#         r"""
#         Update warp based on Bidirectional Composition
#         """
#         self.transform.from_vector_inplace(
#             self.transform.as_vector() +
#             self.alpha * self.dp[:self.n] -
#             self.beta * self.dp[self.n:])
#
#     def run(self, image, initial_shape, gt_shape=None, max_iters=20,
#             map_inference=False, alpha=1.0, beta=1.0):
#         r"""
#         Run Bidirectional Simultaneous algorithms
#         """
#         # set alpha and beta values
#         self.alpha, self.beta = alpha, beta
#         # call super method
#         return super(ABC, self).run(
#             image, initial_shape, gt_shape=gt_shape, max_iters=max_iters,
#             map_inference=map_inference)
#
#
# class MAIC_N(ModifiedAlternating):
#     r"""
#     Simultaneous Inverse Compositional Newton algorithm
#     """
#     def _precompute(self):
#
#         # compute warp jacobian
#         self._dw_dp = self.interface.warp_jacobian()
#
#         # compute U jacobian
#         n_pixels = len(self.template.as_vector()[
#             self.interface.i_mask])
#         self._j_U = np.zeros((self.appearance_model.n_active_components,
#                               n_pixels, self.transform.n_parameters))
#         for k, u in enumerate(self._U.T):
#             self.template2.from_vector_inplace(u)
#             nabla_u = self.interface.gradient(self.template2)
#             j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
#             self._j_U[k, ...] = j_u
#
#         # compute U inverse hessian
#         self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))
#
#     def run(self, image, initial_shape, gt_shape=None, max_iters=20,
#             prior=False):
#
#         # initialize cost
#         cost = []
#         # initialize transform
#         self.transform.set_target(initial_shape)
#         shape_parameters = [self.transform.as_vector()]
#         # initial appearance parameters
#         appearance_parameters = [0]
#         # model mean
#         m = self.appearance_model.mean().as_vector()
#         # masked model mean
#         masked_m = m[self.interface.i_mask]
#
#         for _ in xrange(max_iters):
#
#             # warp image
#             i = self.interface.warp(image)
#             # mask image
#             masked_i = i.as_vector()[self.interface.i_mask]
#
#             if _ == 0:
#                 # project image onto the model bases
#                 c = self._pinv_U.T.dot(masked_i - masked_m)
#             else:
#                 # compute gauss-newton appearance parameters updates
#                 masked_e = (masked_i -
#                             self.template.as_vector()[
#                                 self.interface.i_mask])
#                 dc = (self._pinv_U.T.dot(masked_e - j.dot(dp)) -
#                       masked_e.dot(self._j_U).dot(dp))
#                 c += dc
#
#             # reconstruct appearance
#             t = self._U.dot(c) + m
#             self.template.from_vector_inplace(t)
#             appearance_parameters.append(c)
#
#             # compute error image
#             e = masked_i - masked_m
#
#             # compute model gradient
#             nabla_t = self.interface.gradient(self.template)
#             # compute model second order gradient
#             nabla2_t = self.interface.gradient(Image(nabla_t))
#
#             # compute model jacobian
#             j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
#             # project out appearance model from model jacobian
#             j_po = self.project_out(j)
#
#             # compute gauss-newton hessian
#             h_gn = j_po.T.dot(j)
#             # compute partial newton hessian
#             h_pn = self.interface.partial_newton_hessian(nabla2_t, self._dw_dp)
#             # project out appearance model from error
#             e_po = self.project_out(e)
#             # compute cp hessian
#             h_cp = self._pinv_U.T.dot(j_po) + e_po.dot(self._j_U)
#
#             # compute full newton hessian
#             h = (e_po.dot(h_pn).reshape(h_gn.shape) + h_gn -
#                  h_cp.T.dot(self._inv_h_U.dot(h_cp)))
#             # compute full newton jacobian
#             j = - j_po + self._pinv_U.dot(h_cp)
#
#             # compute gauss-newton parameter updates
#             dp = self.interface.solve(h, j, e, prior)
#
#             # update transform
#             target = self.transform.target
#             self.transform.from_vector_inplace(self.transform.as_vector() + dp)
#             shape_parameters.append(self.transform.as_vector())
#
#             # test convergence
#             # error = np.abs(np.linalg.norm(
#             #     target.points - self.transform.target.points))
#             # if error < self.eps:
#             #     break
#
#             # save cost
#             cost.append(e.T.dot(e))
#
#         # return aam algorithm result
#         return self.interface.algorithm_result(
#             image, shape_parameters, cost,
#             appearance_parameters=appearance_parameters, gt_shape=gt_shape)
#
#
# class MAFC_N(ModifiedAlternating):
#     r"""
#     Simultaneous Forward Compositional Newton algorithm
#     """
#     def _precompute(self):
#
#         # compute warp jacobian
#         self._dw_dp = self.interface.warp_jacobian()
#
#         # compute U jacobian
#         n_pixels = len(self.template.as_vector()[
#             self.interface.i_mask])
#         self._j_U = np.zeros((self.appearance_model.n_active_components,
#                               n_pixels, self.transform.n_parameters))
#         for k, u in enumerate(self._U.T):
#             nabla_u = self.interface.gradient(Image(u.reshape(
#                 self.template.pixels.shape)))
#             j_u = self.interface.steepest_descent_images(nabla_u, self._dw_dp)
#             self._j_U[k, ...] = j_u
#
#         # compute U inverse hessian
#         self._inv_h_U = np.linalg.inv(self._masked_U.T.dot(self._masked_U))
#
#     def run(self, image, initial_shape, gt_shape=None, max_iters=20,
#             prior=False):
#
#         # initialize cost
#         cost = []
#         # initialize transform
#         self.transform.set_target(initial_shape)
#         shape_parameters = [self.transform.as_vector()]
#         # initial appearance parameters
#         appearance_parameters = [0]
#         # model mean
#         m = self.appearance_model.mean().as_vector()
#         # masked model mean
#         masked_m = m[self.interface.i_mask]
#
#         for _ in xrange(max_iters):
#
#             # warp image
#             i = self.interface.warp(image)
#             # mask image
#             masked_i = i.as_vector()[self.interface.i_mask]
#
#             if _ == 0:
#                 # project image onto the model bases
#                 c = self._pinv_U.T.dot(masked_i - masked_m)
#             else:
#                 # compute gauss-newton appearance parameters updates
#                 masked_e = (masked_i -
#                             self.template.as_vector()[
#                                 self.interface.i_mask])
#                 dc = (self._pinv_U.T.dot(masked_e - j.dot(dp)) -
#                       masked_e.dot(self._j_U).dot(dp))
#                 c += dc
#
#             # reconstruct appearance
#             t = self._U.dot(c) + m
#             self.template.from_vector_inplace(t)
#             appearance_parameters.append(c)
#
#             # compute error image
#             e = masked_i - masked_m
#
#             # compute model gradient
#             nabla_i = self.interface.gradient(i)
#             # compute model second order gradient
#             nabla2_i = self.interface.gradient(Image(nabla_i))
#
#             # compute model jacobian
#             j = self.interface.steepest_descent_images(nabla_i, self._dw_dp)
#             # project out appearance model from model jacobian
#             j_po = self.project_out(j)
#
#             # compute gauss-newton hessian
#             h_gn = j_po.T.dot(j)
#             # compute partial newton hessian
#             h_pn = self.interface.partial_newton_hessian(nabla2_i, self._dw_dp)
#             # project out appearance model from error
#             e_po = self.project_out(e)
#             # compute cp hessian
#             h_cp = self._pinv_U.T.dot(j_po) + e_po.dot(self._j_U)
#
#             # compute full newton hessian
#             h = (e_po.dot(h_pn).reshape(h_gn.shape) + h_gn -
#                  h_cp.T.dot(self._inv_h_U.dot(h_cp)))
#             # compute full newton jacobian
#             j = - j_po + self._pinv_U.dot(h_cp)
#
#             # compute gauss-newton parameter updates
#             dp = self.interface.solve(h, j, e, prior)
#
#             # update transform
#             target = self.transform.target
#             self.transform.from_vector_inplace(self.transform.as_vector() + dp)
#             shape_parameters.append(self.transform.as_vector())
#
#             # test convergence
#             # error = np.abs(np.linalg.norm(
#             #     target.points - self.transform.target.points))
#             # if error < self.eps:
#             #     break
#
#             # save cost
#             cost.append(e.T.dot(e))
#
#         # return aam algorithm result
#         return self.interface.algorithm_result(
#             image, shape_parameters, cost,
#             appearance_parameters=appearance_parameters, gt_shape=gt_shape)
