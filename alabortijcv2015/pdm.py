import numpy as np

from menpofit.transform import DifferentiableAlignmentSimilarity
from menpofit.differentiable import DP
from menpofit.modelinstance import ModelInstance, similarity_2d_instance_model


# Point Distribution Models ---------------------------------------------------

class PDM(ModelInstance, DP):
    r"""Specialization of :map:`ModelInstance` for use with spatial data.
    """

    def __init__(self, model, sigma2=1):
        super(PDM, self).__init__(model)
        self._set_prior(sigma2)

    def _set_prior(self, sigma2):
        self.J_prior = sigma2 / self.model.eigenvalues
        self.H_prior = np.diag(self.J_prior)

    @property
    def n_dims(self):
        r"""
        The number of dimensions of the spatial instance of the model

        :type: int
        """
        return self.model.template_instance.n_dims

    def d_dp(self, points):
        """
        Returns the Jacobian of the PCA model reshaped to have the standard
        Jacobian shape:

            n_points    x  n_params      x  n_dims

            which maps to

            n_features  x  n_components  x  n_dims

            on the linear model

        Returns
        -------
        jacobian : (n_features, n_components, n_dims) ndarray
            The Jacobian of the model in the standard Jacobian shape.
        """
        d_dp = self.model.components.reshape(self.model.n_active_components,
                                             -1, self.n_dims)
        return d_dp.swapaxes(0, 1)


class GlobalPDM(PDM):
    r"""
    """
    def __init__(self, model, global_transform_cls, sigma2=1):
        # Start the global_transform as an identity (first call to
        # from_vector_inplace() or set_target() will update this)
        self.global_transform = global_transform_cls(model.mean(),
                                                     model.mean())
        super(GlobalPDM, self).__init__(model, sigma2)

    def _set_prior(self, sigma2):
        sim_prior = np.ones((4,))
        pdm_prior = sigma2 / self.model.eigenvalues
        self.J_prior = np.hstack((sim_prior, pdm_prior))
        self.H_prior = np.diag(self.J_prior)

    @property
    def n_global_parameters(self):
        r"""
        The number of parameters in the `global_transform`

        :type: int
        """
        return self.global_transform.n_parameters

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (`n_global_parameters`,) ndarray
        """
        return self.global_transform.as_vector()

    def _new_target_from_state(self):
        r"""
        Return the appropriate target for the model weights provided,
        accounting for the effect of the global transform


        Returns
        -------

        new_target: :class:`menpo.shape.PointCloud`
            A new target for the weights provided
        """
        return self.global_transform.apply(self.model.instance(self.weights))

    def _weights_for_target(self, target):
        r"""
        Return the appropriate model weights for target provided, accounting
        for the effect of the global transform. Note that this method
        updates the global transform to be in the correct state.

        Parameters
        ----------

        target: :class:`menpo.shape.PointCloud`
            The target that the statistical model will try to reproduce

        Returns
        -------

        weights: (P,) ndarray
            Weights of the statistical model that generate the closest
            PointCloud to the requested target
        """

        self._update_global_transform(target)
        projected_target = self.global_transform.pseudoinverse().apply(target)
        # now we have the target in model space, project it to recover the
        # weights
        new_weights = self.model.project(projected_target)
        # TODO investigate the impact of this, could be problematic
        # the model can't perfectly reproduce the target we asked for -
        # reset the global_transform.target to what it CAN produce
        #refined_target = self._target_for_weights(new_weights)
        #self.global_transform.target = refined_target
        return new_weights

    def _update_global_transform(self, target):
        self.global_transform.set_target(target)

    def _as_vector(self):
        r"""
        Return the current parameters of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (`n_parameters`,) ndarray
            The vector of parameters
        """
        return np.hstack([self.global_parameters, self.weights])

    def from_vector_inplace(self, vector):
        # First, update the global transform
        global_parameters = vector[:self.n_global_parameters]
        self._update_global_weights(global_parameters)
        # Now extract the weights, and let super handle the update
        weights = vector[self.n_global_parameters:]
        PDM.from_vector_inplace(self, weights)

    def _update_global_weights(self, global_weights):
        r"""
        Hook that allows for overriding behavior when the global weights are
        set. Default implementation simply asks global_transform to
        update itself from vector.
        """
        self.global_transform.from_vector_inplace(global_weights)

    def d_dp(self, points):
        # d_dp is always evaluated at the mean shape
        points = self.model.mean().points

        # compute dX/dp

        # dX/dq is the Jacobian of the global transform evaluated at the
        # current target
        # (n_points, n_global_params, n_dims)
        dX_dq = self._global_transform_d_dp(points)

        # by application of the chain rule dX/db is the Jacobian of the
        # model transformed by the linear component of the global transform
        # (n_points, n_weights, n_dims)
        dS_db = PDM.d_dp(self, [])
        # (n_points, n_dims, n_dims)
        dX_dS = self.global_transform.d_dx(points)
        # (n_points, n_weights, n_dims)
        dX_db = np.einsum('ilj, idj -> idj', dX_dS, dS_db)

        # dX/dp is simply the concatenation of the previous two terms
        # (n_points, n_params, n_dims)
        return np.hstack((dX_dq, dX_db))

    def _global_transform_d_dp(self, points):
        return self.global_transform.d_dp(points)


class OrthoPDM(GlobalPDM):
    r"""
    """
    def __init__(self, model, sigma2=1):
        # 1. Construct similarity model from the mean of the model
        self.similarity_model = similarity_2d_instance_model(model.mean())
        # 2. Orthonormalize model and similarity model
        model_cpy = model.copy()
        model_cpy.orthonormalize_against_inplace(self.similarity_model)
        self.similarity_weights = self.similarity_model.project(
            model_cpy.mean())
        super(OrthoPDM, self).__init__(model_cpy,
                                       DifferentiableAlignmentSimilarity,
                                       sigma2)

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (`n_global_parameters`,) ndarray
        """
        return self.similarity_weights

    def _update_global_transform(self, target):
        self.similarity_weights = self.similarity_model.project(target)
        self._update_global_weights(self.similarity_weights)

    def _update_global_weights(self, global_weights):
        self.similarity_weights = global_weights
        new_target = self.similarity_model.instance(global_weights)
        self.global_transform.set_target(new_target)

    def _global_transform_d_dp(self, points):
        return self.similarity_model.components.reshape(
            self.n_global_parameters, -1, self.n_dims).swapaxes(0, 1)
