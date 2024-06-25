# Source: https://gitlab.fit.cvut.cz/kratkeli/ni-dip

# General imports
from typing import Tuple, List, Optional
from sklearn.metrics import recall_score

# src imports
from qsonar.data_objects.project_directories import ProjectDirectories
from qsonar.data_objects.svm_configuration import SVMConfiguration
from qsonar.svm.qsvm import QSVM

# Scikit imports
from sklearn.svm import SVC
from sklearn.metrics import f1_score


class SVM:
	"""
	Implementation of a Support Vector Machine (SVM) with quantum and classical classification methods.
	"""
	def __init__(
		self,
		project_directories: ProjectDirectories,
		config: SVMConfiguration
	):
		"""
		Initialize a SVM object.

		It loads data from a specified dataset and preprocesses it.

		Parameters
		----------
		project_directories : ProjectDirectories
			Configuration for project directories.
		config : SVMConfiguration
			Configuration for classification.
		"""
		self._project_directories = project_directories
		self._config = config
		# load_dataset()

	def load_dataset(self):
		# Load dataset
		# zavolam fci co mi neco vyhodi

		# Set config
		# tyhle čtyři věci v configu potřebuju setnout
		# self._config.train_labels =
		# self._config.test_labels =
		# self._config.train_samples =
		# self._config.test_samples =
		return

	def classify(
		self,
		feature_maps: Optional[List[str]] = None,
		classical_kernels: Optional[List[str]] = None
	) -> None:
		"""
		Perform classification using both quantum and classical methods.

		If feature_maps are provided, it prints the accuracy and F1 score for each quantum feature map.
		If classical_kernels are provided, it prints the accuracy and F1 score for each classical kernel.

		Parameters
		----------
		feature_maps : Optional[List[str]]
			List of quantum feature maps to use for classification. Defaults to None.
		classical_kernels : Optional[List[str]]
			List of classical kernels to use for classification. Defaults to None.

		Returns
		-------
		None

		Examples
		--------
		- svm_instance.classify(feature_maps=["zz", "pauli"], classical_kernels=["linear", "rbf"])
		- svm_instance.classify(feature_maps=["zz", "pauli"])
		- svm_instance.classify(classical_kernels=["linear", "rbf"])
		"""
		print(f'data (train/test): {self._config.train_size}/{self._config.test_size}')

		if feature_maps:
			qsvm = QSVM(self._project_directories, self._config)

			print(f'Number of qubits: {self._config.n_qubits}')
			print(f'Circuits depth: {self._config.depth}\n')

			for feature_map_type in feature_maps:
				print(f'{feature_map_type}FeatureMap')
				accuracy, f1, recall = qsvm.classify(feature_map_type)
				if self._config.runtime_jobs_completed is not False:
					# Either it's on simulator or they finished -> print results
					print(f'accuracy: {accuracy:.3f}')
					print(f'f1: {f1:.3f}')
					print(f'recall: {recall:.3f}\n')

		if classical_kernels:
			for kernel in classical_kernels:
				print(f'{kernel}')
				accuracy, f1, recall = self.__classical_classification(kernel)
				print(f'accuracy: {accuracy:.3f}')
				print(f'f1: {f1:.3f}')
				print(f'recall: {recall:.3f}\n')

	def __classical_classification(
		self,
		classical_kernel: str
	) -> Tuple[float, float]:
		"""
		Perform classical classification using a specified classical kernel.

		Parameters
		----------
		classical_kernel : str
			Type of classical kernel for SVM ("linear", "poly", "rbf", "sigmoid", etc.).

		Returns
		-------
		Tuple[float, float]
			A tuple containing the classification test accuracy and F1 score.
		"""
		classical_svc = SVC(kernel=classical_kernel)

		# Train model
		classical_svc.fit(self._config.train_samples, self._config.train_labels)

		predicted_labels = classical_svc.predict(self._config.test_samples)
		accuracy = classical_svc.score(self._config.test_samples, self._config.test_labels)
		f1 = f1_score(self._config.test_labels, predicted_labels, average='weighted')
		recall = recall_score(self._config.test_labels, predicted_labels)


		return accuracy, f1, recall

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
