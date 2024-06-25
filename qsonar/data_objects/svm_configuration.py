# Source: https://gitlab.fit.cvut.cz/kratkeli/ni-dip

import numpy as np
from dataclasses import dataclass


@dataclass
class SVMConfiguration:
	"""
	Configuration for support vector machine (SVM) classification.

	Attributes
	----------
	train_size : int
		Number of training samples.
	test_size : int
		Number of testing samples.
	n_qubits : int
		Number of qubits to be used in the quantum SVM.
	depth : int
		Depth of the quantum SVM circuit.
	shots : int
		Number of measurement shots to perform when running the quantum circuit. Default is None.
	train_samples : np.ndarray
		Array containing feature vectors of training samples.

			Shape: (n_samples, n_features)

			- n_samples: Number of training samples.
			- n_features: Number of features per sample.
	test_samples : np.ndarray
		Array containing feature vectors of testing samples.

		Shape: (n_samples, n_features)

			- n_samples: Number of testing samples.
			- n_features: Number of features per sample.
	train_labels : np.ndarray
		Array containing labels of training samples.

			Shape: (n_samples,)

			- n_samples: Number of training samples.
	test_labels : np.ndarray
		Array containing labels of testing samples.

		Shape: (n_samples,)

			- n_samples: Number of testing samples.
	simulator: bool
		Whether or not to use the simulator for the classification. Default is True.
	ibm_backend : str
		IBM backend name. Default is None.
	runtime_jobs_completed: bool
		Whether or not IBM Quantum jobs finished. Default is None.
	"""
	train_size: int
	test_size: int
	n_qubits: int
	depth: int
	shots: int = None
	train_samples: np.ndarray = None
	test_samples: np.ndarray = None
	train_labels: np.ndarray = None
	test_labels: np.ndarray = None
	simulator: bool = True
	ibm_backend: str = None
	runtime_jobs_completed: bool = None
