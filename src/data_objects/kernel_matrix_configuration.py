# Source: https://gitlab.fit.cvut.cz/kratkeli/ni-dip

from dataclasses import dataclass
from collections.abc import Sequence
from typing import List, Tuple
from qiskit_ibm_runtime import RuntimeJob

KernelIndices = List[Tuple[int, int]]


@dataclass
class KernelMatrixConfiguration:
	"""
	Configuration for the kernel matrix.

	Attributes
	----------
	is_symmetric: bool
		Indicates whether the kernel matrix is symmetric.
	kernel_shape: Tuple[int, int]
		Shape of the kernel matrix (number of rows, number of columns).
	num_circuits: int
		Total number of circuits used for computing the kernel matrix.
	indices: KernelIndices
		Indices indicating the positions of kernel entries in the matrix.
		Each entry is a tuple (i, j) representing the row index i and column index j.
	jobs: Sequence[RuntimeJob]
		Runtime jobs used for computing kernel entries. Defaults to None.
	jobs_ids: List[str]
		List of jobs IDs. Defaults to None.
	matrix_type : str
		Type of the kernel matrix ("train" or "test"). Defaults to None.
	"""
	is_symmetric: bool
	kernel_shape: Tuple[int, int]
	num_circuits: int
	indices: KernelIndices
	jobs: Sequence[RuntimeJob] = None
	jobs_ids: List[str] = None
	matrix_type: str = None

	def __str__(self):
		"""
		String representation of the KernelMatrixConfiguration object.

		Returns
		-------
		str
			A formatted string.
		"""
		return (
			f"is_symmetric: {self.is_symmetric}\n"
			f"kernel_shape: {self.kernel_shape}\n"
			f"num_circuits: {self.num_circuits}\n"
			f"indices: {self.indices}\n"
			f"jobs: {self.jobs}\n"
			f"jobs_ids: {self.jobs_ids}\n"
			f"matrix_type: {self.matrix_type}\n"
		)
