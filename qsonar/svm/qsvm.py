# Source: https://gitlab.fit.cvut.cz/kratkeli/ni-dip

# General imports
import numpy as np
import pickle
from functools import reduce
from typing import List, Tuple, Optional
import re

# src imports
from qsonar.data_objects.project_directories import ProjectDirectories
from qsonar.data_objects.svm_configuration import SVMConfiguration
from qsonar.data_objects.kernel_matrix_configuration import KernelMatrixConfiguration
from qsonar.svm.fidelity_quantum_kernel_for_qiskit_ibm_runtime import FidelityQuantumKernelForIBMQuantum
from qsonar.svm.compute_uncompute_for_qiskit_ibm_runtime import ComputeUncomputeForIBMQuantum

# Scikit imports
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJob


class QSVM:
	"""
	Implementation of a Quantum Support Vector Machine.
	"""
	def __init__(
		self,
		project_directories: ProjectDirectories,
		config: SVMConfiguration
	):
		"""
		Initialize a QSVM object.

		Parameters
		----------
		project_directories : ProjectDirectories
			Configuration for project directories.
		config : SVMConfiguration
			Configuration for classification.

		Raises
		------
		ValueError
			If the configuration is not valid.
		"""
		self._project_directories = project_directories
		self._config = config

		if self._config.depth is None:
			raise ValueError(f"depth of circuit not provided")
		if self._config.shots is None:
			raise ValueError(f"Shots not provided.")

		if self._config.simulator is False:
			if self._config.runtime_jobs_completed is None:
				raise ValueError(f"Jobs' status not provided")
			if self._project_directories.kernel_matrix_configs_dir is None:
				raise ValueError(f"Path for classification parameters not provided.")

	def classify(
		self,
		feature_map_type: str
	) -> Tuple[Optional[float], Optional[float], Optional[float]]:
		"""
		Perform quantum classification using a specified quantum feature map.

		It retrieves the quantum feature map instance based on the provided type and calculates a Quantum Kernel.
		The quantum kernel is then used for classification.

		Parameters
		----------
		feature_map_type : str
			Type of quantum feature map ("zz", "pauli", "zzPhi", "z").

		Returns
		-------
		Tuple[Optional[float], Optional[float]]
			A tuple containing the classification test accuracy and F1 score.
			If the runtime jobs are not completed, returns (None, None).

		Raises
		------
		ValueError
			If jobs' status or the path for classification parameters is not provided.
		"""
		if self._config.simulator:
			matrix_train, matrix_test = self.__evaluate_on_simulator(feature_map_type)
		else:
			if self._config.runtime_jobs_completed is None:
				raise ValueError(f"Jobs' status not provided.")
			if self._project_directories.kernel_matrix_configs_dir is None:
				raise ValueError(f"Path for classification parameters not provided.")

			matrix_train, matrix_test = self.__evaluate_on_ibm_hardware(feature_map_type)
			# matrix_train or matrix_test are None if the runtime jobs were just submitted
			if matrix_train is None or matrix_test is None:
				return None, None, None

		# Classification
		qsvc = SVC(kernel='precomputed')

		# Train model
		qsvc.fit(matrix_train, self._config.train_labels)

		predicted_labels = qsvc.predict(matrix_test)
		accuracy = qsvc.score(matrix_test, self._config.test_labels)
		f1 = f1_score(self._config.test_labels, predicted_labels, average='weighted')
		recall = recall_score(self._config.test_labels, predicted_labels)

		return accuracy, f1, recall

	def __evaluate_on_simulator(
		self,
		feature_map_type: str
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Evaluate the quantum kernel using a simulator.

		Parameters
		----------
		feature_map_type : str
			Type of quantum feature map circuit to be used for kernel evaluation.

		Returns
		-------
		Tuple[np.ndarray, np.ndarray]
			A tuple containing the kernel matrices for training and testing samples.
		"""
		feature_map = self.__get_feature_map(feature_map_type)

		# Create instance of statevector simulator from qiskit_primitives without WITHOUT NOISE
		sampler = Sampler()
		fidelity = ComputeUncomputeForIBMQuantum(sampler=sampler, shots=self._config.shots)
		kernel = FidelityQuantumKernelForIBMQuantum(feature_map=feature_map, fidelity=fidelity)

		# Compute train matrix
		matrix_train = kernel.evaluate(x_vec=self._config.train_samples)

		# Compute test matrix
		matrix_test = kernel.evaluate(x_vec=self._config.test_samples, y_vec=self._config.train_samples)

		return matrix_train, matrix_test

	def __evaluate_on_ibm_hardware(
		self,
		feature_map_type: str
	) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
		"""
		Evaluate the quantum kernel using IBM Quantum hardware.

		Parameters
		----------
		feature_map_type : str
			Type of quantum feature map circuit to be used for kernel evaluation.

		Returns
		-------
		Tuple[Optional[np.ndarray], Optional[np.ndarray]]
			Kernel matrices for training and testing samples, respectively.
			If the runtime jobs are not completed, returns (None, None).
		"""
		if self._config.runtime_jobs_completed is False:
			self.__submit_jobs(feature_map_type)
			return None, None
		else:
			return self.__process_finished_jobs(feature_map_type)

	def __submit_jobs(
		self,
		feature_map_type: str
	) -> None:
		"""
		Submit jobs to calculate the kernel matrices using the specified quantum feature map.

		This method submits jobs to calculate the kernel matrices for training and testing samples
		using the specified quantum feature map. The jobs are submitted to the IBM Quantum Runtime
		service for execution.

		Parameters
		----------
		feature_map_type : str
			Type of quantum feature map ("zz", "pauli", "zzPhi", "z").

		Returns
		-------
		None

		Raises
		------
		ValueError
			If the IBM backend is not provided.
		"""
		if self._config.ibm_backend is None:
			raise ValueError(f"IBMBackend not provided.")
		feature_map = self.__get_feature_map(feature_map_type)
		# Start service
		service = QiskitRuntimeService()

		# Run on the provided backend
		# backend = service.backend(self._config.ibm_backend)
		backend = service.least_busy(simulator=False, operational=True, min_num_qubits=4)
		re_backend = re.match(r"\<IBMBackend\('([^']+)'\)\>", str(backend)).group(1)
		self._config.ibm_backend = str(re_backend)

		fidelity = ComputeUncomputeForIBMQuantum(backend=backend, simulator=False, shots=self._config.shots)
		kernel = FidelityQuantumKernelForIBMQuantum(feature_map=feature_map, fidelity=fidelity)

		# Submit train jobs
		matrix_config = kernel.get_kernel_matrix_config(self._config.train_samples)
		matrix_config.matrix_type = "train"
		# Save train parameters
		self.__save_matrix_config(feature_map_type, matrix_config)

		# Submit test jobs
		matrix_config = kernel.get_kernel_matrix_config(self._config.test_samples, self._config.train_samples)
		matrix_config.matrix_type = "test"
		# Save test parameters
		self.__save_matrix_config(feature_map_type, matrix_config)

		print("Waiting for jobs to finish.")

	def __process_finished_jobs(
		self,
		feature_map_type: str
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Process the finished jobs to obtain the kernel matrices.

		This method processes the finished jobs to obtain the kernel matrices for training and testing
		samples using the specified quantum feature map. It loads the kernel matrix configurations,
		retrieves the necessary information, and computes the kernel matrices.

		The method assumes that the jobs for computing the kernel matrices have already finished
		execution and the results are available.

		Parameters
		----------
		feature_map_type : str
			Type of quantum feature map ("zz", "pauli", "zzPhi", "z").

		Returns
		-------
		Tuple[np.ndarray, np.ndarray]
			Kernel matrices for training and testing samples.
		"""
		feature_map = self.__get_feature_map(feature_map_type)
		# Start service
		service = QiskitRuntimeService()

		fidelity = ComputeUncomputeForIBMQuantum(simulator=False)
		kernel = FidelityQuantumKernelForIBMQuantum(feature_map=feature_map, fidelity=fidelity)

		# Load train matrix config
		matrix_type = "train"
		matrix_config = self.__load_matrix_configuration(feature_map_type, service, matrix_type)

		# Compute train matrix
		matrix_train = kernel.evaluate_matrix(matrix_config)

		# Load test matrix config
		matrix_type = "test"
		matrix_config = self.__load_matrix_configuration(feature_map_type, service, matrix_type)

		# Comopute test matrix
		matrix_test = kernel.evaluate_matrix(matrix_config)

		return matrix_train, matrix_test

	def __save_matrix_config(
		self,
		feature_map_type: str,
		matrix_config: KernelMatrixConfiguration
	) -> None:
		"""
		Save the kernel matrix configuration to a file.

		The method serializes the `matrix_config` object using pickle and saves it to a file
		with a specified naming convention based on the feature map type, matrix type, and dataset sizes.
		The file is stored in the directory containing configurations of kernel matrices within the dataset directory.

		If an error occurs during file writing, appropriate error messages are printed.

		Parameters
		----------
		feature_map_type : str
			Type of the quantum feature map used for generating the kernel matrix.
		matrix_config : KernelMatrixConfiguration
			Configuration of the kernel matrix to be saved.

		Raises
		------
		ValueError
			If the `matrix_type` parameter is neither "train" nor "test".

		Returns
		-------
		None
		"""
		if matrix_config.matrix_type != "train" and matrix_config.matrix_type != "test":
			raise ValueError(f"Wrong matrix type")

		matrix_config.jobs_ids = []
		for job in matrix_config.jobs:
			matrix_config.jobs_ids.append(job.job_id())

		matrix_config.jobs = None

		file_name = f'{feature_map_type}FeatureMap_{matrix_config.matrix_type}_matrix_{self._config.train_size}_train_{self._config.test_size}_test_{self._config.ibm_backend}_backend.pkl'
		file_path = (
				self._project_directories.working_dir +
				self._project_directories.dataset_dir +
				self._project_directories.kernel_matrix_configs_dir +
				file_name
		)

		try:
			with open(file_path, 'wb') as f:
				pickle.dump(matrix_config, f)
		except PermissionError:
			print("Permission denied.")
		except OSError as e:
			print(f"Error: Unable to write to file - {e}")
		except Exception as e:
			print(f"Unexpected error: {e}")

	def __load_matrix_configuration(
		self,
		feature_map_type: str,
		service: QiskitRuntimeService,
		matrix_type: str
	) -> KernelMatrixConfiguration:
		"""
		Load the kernel matrix configuration from file.

		This method loads the kernel matrix configuration from a saved file for a specific type
		of quantum feature map and matrix type (train or test). It retrieves the file path based on
		the provided parameters, reads the file, and returns the loaded kernel matrix configuration.

		The method assumes that the kernel matrix configuration has been previously saved to a file
		with the appropriate file name format.

		Parameters
		----------
		feature_map_type : str
			Type of quantum feature map ("zz", "pauli", "zzPhi", "z").
		service : QiskitRuntimeService
			Qiskit runtime service instance for retrieving job information.
		matrix_type : str
			Type of matrix ("train" or "test").

		Returns
		-------
		KernelMatrixConfiguration
			Loaded kernel matrix configuration.

		Raises
		------
		ValueError
			If the provided matrix type is invalid.
		"""
		if matrix_type != "train" and matrix_type != "test":
			raise ValueError(f"Wrong matrix type")

		file_name = f'{feature_map_type}FeatureMap_{matrix_type}_matrix_{self._config.train_size}_train_{self._config.test_size}_test_{self._config.ibm_backend}_backend.pkl'
		full_path = (
				self._project_directories.working_dir +
				self._project_directories.dataset_dir +
				self._project_directories.kernel_matrix_configs_dir +
				file_name
		)

		try:
			with open(full_path, 'rb') as f:
				matrix_config = pickle.load(f)
		except FileNotFoundError as e:
			print(f"Error: File not found - {e}")
		except PermissionError:
			print("Permission denied.")
		except OSError as e:
			print(f"Error: Unable to read file - {e}")
		except Exception as e:
			print(f"Unexpected error: {e}")

		matrix_config.jobs = self.__load_jobs(service, matrix_config.jobs_ids)
		return matrix_config

	@staticmethod
	def __load_jobs(
		service: QiskitRuntimeService,
		jobs_ids: List[str]
	) -> List[RuntimeJob]:
		"""
		Load runtime jobs from Qiskit runtime service.

		This method loads runtime jobs from the Qiskit runtime service based on the provided job IDs.
		It retrieves each job from the service and appends it to a list of loaded jobs, which is then returned.

		The method assumes that the job IDs provided correspond to existing runtime jobs in the Qiskit runtime service.

		Parameters
		----------
		service : QiskitRuntimeService
			Qiskit runtime service instance for retrieving job information.
		jobs_ids : List[str]
			List of jobs IDs representing the jobs to be loaded.

		Returns
		-------
		List[RuntimeJob]
			List of loaded runtime jobs.
		"""
		loaded_jobs = []
		for job_id in jobs_ids:
			job = service.job(job_id)
			loaded_jobs.append(job)
		return loaded_jobs

	@staticmethod
	def __custom_data_map_func(x: np.ndarray) -> float:
		"""
		Custom data mapping function for feature map coefficients calculation.

		Parameters
		----------
		x : np.ndarray
			Input data for mapping.

		Returns
		-------
		float
			Coefficient calculated based on the input data.
		"""
		coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, np.sin(np.pi - x))
		return coeff

	def __get_feature_map(
			self,
			feature_map_type: str
	) -> QuantumCircuit:
		"""
		Get a quantum feature map based on the specified type.

		Supported types:
		- "zz": ZZFeatureMap with linear entanglement and inserted barriers.
		- "pauli": PauliFeatureMap with specified paulis ('X', 'Y', 'ZZ') and linear entanglement.
		- "zzPhi": PauliFeatureMap with specified paulis ('Z', 'ZZ') and a custom data mapping function.
		- "z": ZFeatureMap with linear entanglement.

		Parameters
		----------
		feature_map_type : str
			Type of quantum feature map ("zz", "pauli", "zzPhi", "z").

		Returns
		-------
		Quantum feature map instance based on the specified type.

		Raises
		------
		ValueError
			If an unsupported feature map type is provided.
		"""
		if feature_map_type == "zz":
			return ZZFeatureMap(
				feature_dimension=self._config.n_qubits,
				reps=self._config.depth,
				entanglement='linear',
				insert_barriers=True
			)
		elif feature_map_type == "pauli":
			return PauliFeatureMap(
				feature_dimension=self._config.n_qubits,
				reps=self._config.depth,
				paulis=['X', 'Y', 'ZZ']
			)
		elif feature_map_type == "zzPhi":
			return PauliFeatureMap(
				feature_dimension=self._config.n_qubits,
				reps=self._config.depth,
				paulis=['Z', 'ZZ'],
				data_map_func=self.__custom_data_map_func
			)
		elif feature_map_type == "z":
			return ZFeatureMap(
				feature_dimension=self._config.n_qubits,
				reps=self._config.depth,
			)
		else:
			raise ValueError(f"Unsupported feature map type: {feature_map_type}")

	@property
	def configuration(self) -> SVMConfiguration:
		"""
		Get configuration.

		Returns
		-------
		SVMConfiguration
			Configuration for support vector machine (SVM) classification.
		"""
		return self._config
