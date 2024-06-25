# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# This code is a modified version of the Fidelity Quantum Kernel.
"""Fidelity Quantum Kernel compatible with IBM Quantum"""

# Source: https://gitlab.fit.cvut.cz/kratkeli/ni-dip

from __future__ import annotations

# General imports
from collections.abc import Sequence
from typing import List, Tuple, Optional
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_machine_learning.kernels import BaseKernel
from qiskit_ibm_runtime import RuntimeJob

# src imports
from qsonar.svm.compute_uncompute_for_qiskit_ibm_runtime import ComputeUncomputeForIBMQuantum
from qsonar.data_objects.kernel_matrix_configuration import KernelMatrixConfiguration

KernelIndices = List[Tuple[int, int]]


class FidelityQuantumKernelForIBMQuantum(BaseKernel):
    r"""
    Fidelity Quantum Kernel implementation for IBM Quantum.

    This class is a modified version of the original FidelityQuantumKernel class
    from the Qiskit Machine Learning module. It includes custom modifications and
    additions tailored for use with IBM Quantum systems.

    An implementation of the quantum kernel interface based on the
    :class:`~qiskit_algorithms.state_fidelities.BaseStateFidelity` algorithm.

    Here, the kernel function is defined as the overlap of two quantum states defined by a
    parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2
    """

    def __init__(
        self,
        *,
        feature_map: Optional[QuantumCircuit] = None,
        fidelity: ComputeUncomputeForIBMQuantum,
        enforce_psd: bool = True,
        evaluate_duplicates: str = "off_diagonal",
        max_circuits_per_job: int = None,
    ) -> None:
        """
        Initialize a FidelityQuantumKernelForQiskitIBMRuntime object.

        Parameters
        ----------
        feature_map:
            Parameterized circuit to be used as the feature map. If ``None`` is given,
            :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
            a mismatch in the number of qubits of the feature map and the number of features
            in the dataset, then the kernel will try to adjust the feature map to reflect the
            number of features.
        fidelity: ComputeUncomputeForQiskitIBMRuntime
            An instance of the ComputeUncomputeForQiskitIBMRuntime primitive to be used
            to compute fidelity between states. It is a modified version of the
            :class:`~qiskit_algorithms.state_fidelities.ComputeUncompute` and it is intended
            to run on IBM Quantum system.
        enforce_psd: bool
            Project to the closest positive semidefinite matrix if ``x = y``.
            Default ``True``.
        evaluate_duplicates: str
            Defines a strategy how kernel matrix elements are evaluated if
            duplicate samples are found. Possible values are:

                - ``all`` means that all kernel matrix elements are evaluated, even the diagonal
                  ones when training. This may introduce additional noise in the matrix.
                - ``off_diagonal`` when training the matrix diagonal is set to `1`, the rest
                  elements are fully evaluated, e.g., for two identical samples in the
                  dataset. When inferring, all elements are evaluated. This is the default
                  value.
                - ``none`` when training the diagonal is set to `1` and if two identical samples
                  are found in the dataset the corresponding matrix element is set to `1`.
                  When inferring, matrix elements for identical samples are set to `1`.
        max_circuits_per_job: int
            Maximum number of circuits per job for the backend. Please
            check the backend specifications. Use ``None`` for all entries per job.
            Default ``None``.

        Raises
        ------
            ValueError
                When unsupported value is passed to `evaluate_duplicates` or to `max_circuits_per_job`.
        """
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)

        eval_duplicates = evaluate_duplicates.lower()
        if eval_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Unsupported value passed as evaluate_duplicates: {evaluate_duplicates}"
            )
        self._evaluate_duplicates = eval_duplicates
        self._fidelity = fidelity
        if max_circuits_per_job is not None:
            if max_circuits_per_job < 1:
                raise ValueError(
                    f"Unsupported value passed as max_circuits_per_job: {max_circuits_per_job}"
                )
        self._max_circuits_per_job = max_circuits_per_job

    def get_kernel_matrix_config(
        self,
        x_vec: np.ndarray,
        y_vec: Optional[np.ndarray] = None
    ) -> KernelMatrixConfiguration:
        """
        Get the configuration for computing the kernel matrix entries by executing the
        underlying fidelity instance.

        If y_vec is None, self inner product is calculated.

        Parameters
        ----------
        x_vec : np.ndarray
            1D or 2D array of datapoints, NxD, where N is the number of datapoints,
            D is the feature dimension.
        y_vec : Optional[np.ndarray], optional
            1D or 2D array of datapoints, MxD, where M is the number of datapoints,
            D is the feature dimension.

        Returns
        -------
        KernelMatrixConfiguration
            The configuration for computing the kernel matrix entries, including the
            jobs with the corresponding parameters.
        """
        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])
        if is_symmetric:
            left_parameters, right_parameters, indices = self.__get_symmetric_parameterization(x_vec)
            jobs = self.__run_jobs(left_parameters, right_parameters)
        else:
            left_parameters, right_parameters, indices = self.__get_parameterization(x_vec, y_vec)
            jobs = self.__run_jobs(left_parameters, right_parameters)

        num_circuits = left_parameters.shape[0]
        matrix_config = KernelMatrixConfiguration(is_symmetric, kernel_shape, num_circuits, indices, jobs)
        # print(matrix_config)
        return matrix_config

    def evaluate_matrix(
        self,
        matrix_config: KernelMatrixConfiguration
    ) -> np.ndarray:
        """
        Construct the kernel matrix for the given data using the provided matrix configuration.

        Parameters
        ----------
        matrix_config : KernelMatrixConfiguration
            The configuration containing the necessary parameters for computing the kernel matrix,
            including the shape, number of circuits, indices, and jobs.

        Returns
        -------
        np.ndarray
            2D kernel matrix, NxM.
        """
        if matrix_config.is_symmetric:
            kernel_matrix = self.__get_symmetric_kernel_matrix(matrix_config)
        else:
            kernel_matrix = self.__get_kernel_matrix(matrix_config)

        if matrix_config.is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)

        return kernel_matrix

    def evaluate(
        self,
        x_vec: np.ndarray,
        y_vec: np.ndarray | None = None
    ) -> np.ndarray:
        matrix_config = self.get_kernel_matrix_config(x_vec, y_vec)
        kernel_matrix = self.evaluate_matrix(matrix_config)
        return kernel_matrix

    def __get_parameterization(
        self,
        x_vec: np.ndarray,
        y_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, KernelIndices]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = []
        for i, x_i in enumerate(x_vec):
            for j, y_j in enumerate(y_vec):
                if self.__is_trivial(i, j, x_i, y_j, False):
                    continue

                left_parameters = np.vstack((left_parameters, x_i))
                right_parameters = np.vstack((right_parameters, y_j))
                indices.append((i, j))

        return left_parameters, right_parameters, indices

    def __get_symmetric_parameterization(
        self,
        x_vec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, KernelIndices]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = []
        for i, x_i in enumerate(x_vec):
            for j, x_j in enumerate(x_vec[i:]):
                if self.__is_trivial(i, i + j, x_i, x_j, True):
                    continue

                left_parameters = np.vstack((left_parameters, x_i))
                right_parameters = np.vstack((right_parameters, x_j))
                indices.append((i, i + j))

        return left_parameters, right_parameters, indices

    def __get_kernel_matrix(
        self,
        matrix_config: KernelMatrixConfiguration
    ) -> np.ndarray:
        """
        Given a parameterization, this computes the symmetric kernel matrix.
        """
        kernel_entries = self.__get_kernel_entries(matrix_config)

        # fill in trivial entries and then update with fidelity values
        kernel_matrix = np.ones(matrix_config.kernel_shape)

        for i, (col, row) in enumerate(matrix_config.indices):
            kernel_matrix[col, row] = kernel_entries[i]

        return kernel_matrix

    def __get_symmetric_kernel_matrix(
        self,
        matrix_config: KernelMatrixConfiguration
    ) -> np.ndarray:
        """
        Given a set of parameterization, this computes the kernel matrix.
        """
        kernel_entries = self.__get_kernel_entries(matrix_config)
        kernel_matrix = np.ones(matrix_config.kernel_shape)

        for i, (col, row) in enumerate(matrix_config.indices):
            kernel_matrix[col, row] = kernel_entries[i]
            kernel_matrix[row, col] = kernel_entries[i]

        return kernel_matrix

    def __get_kernel_entries(
        self,
        matrix_config: KernelMatrixConfiguration
    ) -> Sequence[float]:
        """
        Gets kernel entries by executing the underlying fidelity instance and getting the results
        back from the async job.
        """
        matrix_config.kernel_entries = []
        # Check if it is trivial case, only identical samples
        if matrix_config.num_circuits != 0:
            if self._max_circuits_per_job is None:
                matrix_config.kernel_entries = self._fidelity.get_fidelities(matrix_config.jobs)
            else:
                # Determine the number of chunks needed
                num_chunks = (
                    matrix_config.num_circuits + self._max_circuits_per_job - 1
                ) // self._max_circuits_per_job
                for i in range(num_chunks):
                    # Extend the kernel_entries list with the results from this chunk
                    matrix_config.kernel_entries.extend(self._fidelity.get_fidelities(matrix_config.jobs))
        return matrix_config.kernel_entries

    def __run_jobs(self, left_parameters: np.ndarray, right_parameters: np.ndarray) -> Sequence[RuntimeJob]:
        """
           Executes the underlying fidelity instance.
        """
        num_circuits = left_parameters.shape[0]
        jobs = []
        if num_circuits != 0:
            if self._max_circuits_per_job is None:
                jobs = self._fidelity.run_jobs(
                    [self._feature_map] * num_circuits,
                    [self._feature_map] * num_circuits,
                    left_parameters,
                    right_parameters,
                )
            else:
                # Determine the number of chunks needed
                num_chunks = (
                    num_circuits + self._max_circuits_per_job - 1
                ) // self._max_circuits_per_job
                for i in range(num_chunks):
                    # Determine the range of indices for this chunk
                    start_idx = i * self._max_circuits_per_job
                    end_idx = min((i + 1) * self._max_circuits_per_job, num_circuits)
                    # Extract the parameters for this chunk
                    chunk_left_parameters = left_parameters[start_idx:end_idx]
                    chunk_right_parameters = right_parameters[start_idx:end_idx]
                    # Execute this chunk
                    jobs = self._fidelity.run_jobs(
                        [self._feature_map] * (end_idx - start_idx),
                        [self._feature_map] * (end_idx - start_idx),
                        chunk_left_parameters,
                        chunk_right_parameters,
                    )
        return jobs

    def __is_trivial(
        self, i: int, j: int, x_i: np.ndarray, y_j: np.ndarray, symmetric: bool
    ) -> bool:
        """
        Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

        Args:
            i: row index of the entry in the kernel matrix.
            j: column index of the entry in the kernel matrix.
            x_i: a sample from the dataset that corresponds to the row in the kernel matrix.
            y_j: a sample from the dataset that corresponds to the column in the kernel matrix.
            symmetric: whether it is a symmetric case or not.

        Returns:
            `True` if the entry is trivial, `False` otherwise.
        """
        # if we evaluate all combinations, then it is non-trivial
        if self._evaluate_duplicates == "all":
            return False

        # if we are on the diagonal and we don't evaluate it, it is trivial
        if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
            return True

        # if don't evaluate any duplicates
        if np.array_equal(x_i, y_j) and self._evaluate_duplicates == "none":
            return True

        # otherwise evaluate
        return False

    @property
    def fidelity(self):
        """Returns the fidelity primitive used by this kernel."""
        return self._fidelity

    @property
    def evaluate_duplicates(self):
        """Returns the strategy used by this kernel to evaluate kernel matrix elements if duplicate
        samples are found."""
        return self._evaluate_duplicates
