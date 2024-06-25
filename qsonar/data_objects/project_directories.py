# Source: https://gitlab.fit.cvut.cz/kratkeli/ni-dip

from dataclasses import dataclass


@dataclass
class ProjectDirectories:
	"""
	Configuration for project directories.

	Attributes
	----------
	working_dir : str
		The absolute path to the working directory in which the dataset is stored.
	dataset_dir : str
		The relative path to the dataset directory within the working directory.
		Defaults to "pe-machine-learning-dataset/".
	kernel_matrix_configs_dir : str
		Relative path to the directory containing configurations of kernel matrices.
		Defaults to "kernel-matrix-configs/".
	results_dir : str
		Relative path to the directory containing classification results.
		Defaults to "results/".
	"""
	working_dir: str
	dataset_dir: str = ""
	kernel_matrix_configs_dir: str = "kernel-matrix-configs/"
	results_dir: str = "results/"

	def __str__(self):
		"""
		String representation of the ProjectDirectories object.

		Returns
		-------
		str
			A formatted string containing the paths of the project directories.
		"""
		return (
			f"working_dir: {self.working_dir}\n"
			f"dataset_dir: {self.dataset_dir}\n"
			f"kernel_matrix_configs_dir: {self.kernel_matrix_configs_dir}\n"
			f"results_dir: {self.results_dir}\n"
		)
