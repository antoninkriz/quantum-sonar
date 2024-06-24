# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# This code is a modified version of the Compute-uncompute fidelity interface.
"""
Compute-uncompute fidelity interface compatible with IBM Quantum
"""
from __future__ import annotations

# Source: https://gitlab.fit.cvut.cz/kratkeli/ni-dip

# General imports
from collections.abc import Sequence
from copy import copy
from typing import Optional, List

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import Options, Backend
from qiskit.circuit import ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.state_fidelities import BaseStateFidelity
from qiskit_algorithms import AlgorithmJob
from qiskit_ibm_runtime import Session, Sampler, RuntimeJob


class ComputeUncomputeForIBMQuantum(BaseStateFidelity):
    r"""
    ComputeUncompute implementation for IBM Quantum.

    This class is a modified version of the original ComputeUncompute class
    from the Qiskit Machine Learning module. It includes custom modifications and
    additions tailored for use with IBM Quantum systems.

    This class leverages the sampler primitive to calculate the state
    fidelity of two quantum circuits following the compute-uncompute
    method (see [1] for further reference).

    The implementation is a modified version of the ComputeUncompute
    and is intended to run on IBM Quantum systems.

    The fidelity can be defined as the state overlap.

    .. math::

            |\langle\psi(x)|\phi(y)\rangle|^2

    where :math:`x` and :math:`y` are optional parametrizations of the
    states :math:`\psi` and :math:`\phi` prepared by the circuits
    ``circuit_1`` and ``circuit_2``, respectively.

    **Reference:**
    [1] Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala,
    A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning
    with quantum-enhanced feature spaces. Nature, 567(7747), 209-212.
    `arXiv:1804.11326v2 [quant-ph] <https://arxiv.org/pdf/1804.11326.pdf>`_

    """

    def __init__(
        self,
        sampler: Optional[BaseSampler] = None,
        simulator: bool = True,
        backend: Optional[Backend] = None,
        shots: int = None,
        options: Optional[Options] = None,
        local: bool = False,
    ) -> None:
        r"""
        Initialize a ComputeUncomputeForQiskitIBMRuntime object.

        Parameters
        ----------
        sampler : Optional[BaseSampler]
            Sampler primitive instance compatible with IBM Quantum. Defaults to None.
        backend : Optional[Backend]
            Backend instance of IBM Quantum system. Defaults to None.
        options : Optional[Options]
            Primitive backend runtime options used for circuit execution.
            The order of priority is: options in ``run`` method > fidelity's
            default options > primitive's default setting.
            Higher priority setting overrides lower priority setting.
            Defaults to None.
        local : bool
            If set to ``True``, the fidelity is averaged over
            single-qubit projectors

            .. math::

                \hat{O} = \frac{1}{N}\sum_{i=1}^N|0_i\rangle\langle 0_i|,

            instead of the global projector :math:`|0\rangle\langle 0|^{\otimes n}`.
            This coincides with the standard (global) fidelity in the limit of
            the fidelity approaching 1. Might be used to increase the variance
            to improve trainability in algorithms such as :class:`~.time_evolvers.PVQD`.

        Raises
        ------
            ValueError
                If the sampler is not an instance of ``BaseSampler``.
        """
        if simulator:
            if sampler is None:
                raise ValueError(f"Sampler not provided")
            if not isinstance(sampler, BaseSampler):
                raise ValueError(
                    f"The sampler should be an instance of BaseSampler, " f"but got {type(sampler)}"
                )
        self._sampler: BaseSampler = sampler
        self._simulator = simulator
        self._backend = backend
        self._shots = shots
        self._local = local
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        super().__init__()

    def create_fidelity_circuit(
        self, circuit_1: QuantumCircuit, circuit_2: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Combines ``circuit_1`` and ``circuit_2`` to create the
        fidelity circuit following the compute-uncompute method.

        Args:
            circuit_1: (Parametrized) quantum circuit.
            circuit_2: (Parametrized) quantum circuit.

        Returns:
            The fidelity quantum circuit corresponding to circuit_1 and circuit_2.
        """
        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()

        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit

    def __run_on_simulator(
        self,
        circuits: Sequence[QuantumCircuit],
        values: List[List[float]]
    ) -> Sequence[[RuntimeJob]]:
        """
        Run the circuits on a quantum simulator.

        Parameters
        ----------
        circuits : Sequence[QuantumCircuit]
            List of quantum circuits to be executed on the simulator.
        values : List[List[float]]
            List of parameter values for each circuit in `circuits`.

        Returns
        -------
        Sequence[RuntimeJob]
            List of runtime jobs representing the execution of the circuits on the simulator.
        """
        sampler_job = self._sampler.run(circuits=circuits, parameter_values=values, shots=self._shots)
        return [sampler_job]

    def __run_on_ibm_hardware(
        self,
        circuits: Sequence[QuantumCircuit],
        values: List[List[float]]
    ) -> Sequence[[RuntimeJob]]:
        """
        Run the circuits on IBM Quantum hardware.

        Parameters
        ----------
        circuits : Sequence[QuantumCircuit]
            List of quantum circuits to be executed on IBM Quantum hardware.
        values : List[List[float]]
            List of parameter values for each circuit in `circuits`.

        Returns
        -------
        Sequence[RuntimeJob]
            List of runtime jobs representing the execution of the circuits on IBM Quantum hardware.
        """
        jobs = []

        with Session(backend=self._backend) as session:
            sampler = Sampler(session=session)
            for i, circuit in enumerate(circuits):
                sampler_job = sampler.run(circuits=circuit, parameter_values=values[i], shots=self._shots)
                jobs.append(sampler_job)

        print("Jobs submitted!")
        return jobs

    def run_jobs(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **options,
    ) -> Sequence[[RuntimeJob]]:
        r"""
        Submit jobs for computing the state overlap (fidelity) calculation between two
        (parametrized) circuits (first and second) for a specific set of parameter
        values (first and second) following the compute-uncompute method.

        Parameters
        ----------
        circuits_1: QuantumCircuit or Sequence[QuantumCircuit]
            The first (parametrized) quantum circuits preparing :math:`|\psi\rangle`.
        circuits_2: QuantumCircuit or Sequence[QuantumCircuit]
            The second (parametrized) quantum circuits preparing :math:`|\psi\rangle`.
        values_1: Sequence[float] or Sequence[Sequence[float]] or None, optional
            Numerical parameters to be bound to the first circuits.
        values_2: Sequence[float] or Sequence[Sequence[float]] or None, optional
            Numerical parameters to be bound to the second circuits.
        options: dict
            Primitive backend runtime options used for circuit execution.
            The order of priority is: options in the `run` method > fidelity's
            default options > primitive's default setting.
            Higher priority setting overrides lower priority setting.

        Returns
        -------
        Sequence[RuntimeJob]
            Jobs for the fidelity calculation.

        Raises
        ------
        ValueError
            If at least one pair of circuits is not defined.
        """
        circuits = self._construct_circuits(circuits_1, circuits_2)
        if len(circuits) == 0:
            raise ValueError(
                "At least one pair of circuits must be defined to calculate the state overlap."
            )
        values = self._construct_value_list(circuits_1, circuits_2, values_1, values_2)

        # The priority of run options is as follows:
        # options in `evaluate` method > fidelity's default options >
        # primitive's default options.
        opts = copy(self._default_options)
        opts.update_options(**options)

        if self._simulator:
            return self.__run_on_simulator(circuits, values)
        else:
            return self.__run_on_ibm_hardware(circuits, values)

    def get_fidelities(
        self,
        jobs: Sequence[RuntimeJob],
    ) -> Sequence[float]:
        """
        Process results from the submitted jobs and calculate fidelities.

        Parameters
        ----------
        jobs : Sequence[RuntimeJob]
            Sequence of runtime jobs.

        Returns
        -------
        Sequence[float]
            List of fidelities.
        """
        if self._simulator:
            job = jobs[0]
            result = job.result()
            raw_fidelities = [
                ComputeUncomputeForIBMQuantum._get_global_fidelity(prob_dist) for prob_dist in result.quasi_dists
            ]
            return ComputeUncomputeForIBMQuantum._truncate_fidelities(raw_fidelities)
        else:
            fidelities = []
            for job in jobs:
                result = job.result()
                raw_fidelity = [ComputeUncomputeForIBMQuantum._get_global_fidelity(prob_dist) for prob_dist in result.quasi_dists]
                fidelity = ComputeUncomputeForIBMQuantum._truncate_fidelities(raw_fidelity)
                fidelities.append(fidelity[0])
            return fidelities

    def _construct_circuits(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
    ) -> Sequence[QuantumCircuit]:
        """
        Constructs the list of fidelity circuits to be evaluated.
        These circuits represent the state overlap between pairs of input circuits,
        and their construction depends on the fidelity method implementations.
        Circuits are transpiled for the IBM Quantum backend.

        Args:
            circuits_1: (Parametrized) quantum circuits.
            circuits_2: (Parametrized) quantum circuits.

        Returns:
            List of constructed fidelity circuits.

        Raises:
            ValueError: if the length of the input circuit lists doesn't match.
        """

        if isinstance(circuits_1, QuantumCircuit):
            circuits_1 = [circuits_1]
        if isinstance(circuits_2, QuantumCircuit):
            circuits_2 = [circuits_2]

        if len(circuits_1) != len(circuits_2):
            raise ValueError(
                f"The length of the first circuit list({len(circuits_1)}) "
                f"and second circuit list ({len(circuits_2)}) is not the same."
            )

        circuits = []

        for (circuit_1, circuit_2) in zip(circuits_1, circuits_2):

            # Use the same key for circuits as qiskit.primitives use.
            circuit = self._circuit_cache.get((_circuit_key(circuit_1), _circuit_key(circuit_2)))

            if circuit is not None:
                circuits.append(circuit)
            else:
                self._check_qubits_match(circuit_1, circuit_2)

                # re-parametrize input circuits
                # TODO: make smarter checks to avoid unnecessary re-parametrizations
                parameters_1 = ParameterVector("x", circuit_1.num_parameters)
                parametrized_circuit_1 = circuit_1.assign_parameters(parameters_1)
                parameters_2 = ParameterVector("y", circuit_2.num_parameters)
                parametrized_circuit_2 = circuit_2.assign_parameters(parameters_2)

                circuit = self.create_fidelity_circuit(
                    parametrized_circuit_1, parametrized_circuit_2
                )
                self.fidelity_circuit = circuit

                if not self._simulator:
                    if self._backend is None:
                        raise ValueError(f"IBMBackend not provided")
                    # Generate ISA circuit
                    pm = generate_preset_pass_manager(backend=self._backend, optimization_level=1)
                    circuit = pm.run(circuit)
                    self.transpiled_circuit = circuit

                circuits.append(circuit)
                # update cache
                self._circuit_cache[_circuit_key(circuit_1), _circuit_key(circuit_2)] = circuit

        return circuits

    def _run(
        self,
        circuits_1: QuantumCircuit | Sequence[QuantumCircuit],
        circuits_2: QuantumCircuit | Sequence[QuantumCircuit],
        values_1: Sequence[float] | Sequence[Sequence[float]] | None = None,
        values_2: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **options,
    ) -> AlgorithmJob:
        """
        Dummy method.
        """
        return AlgorithmJob(ComputeUncomputeForIBMQuantum._call)

    def _call(self):
        """
        Dummy method.
        """
        return

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and fidelity default options,
        where, if the same field is set in both, the fidelity's default options override
        the primitive's default setting.

        Returns:
            The fidelity default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the fidelity's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the fidelity default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > fidelity's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The fidelity default + estimator + run options.
        """
        opts = copy(self._sampler.options)
        opts.update_options(**options)
        return opts

    @staticmethod
    def _get_global_fidelity(probability_distribution: dict[int, float]) -> float:
        """Process the probability distribution of a measurement to determine the
        global fidelity.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The global fidelity.
        """
        return probability_distribution.get(0, 0)

    @staticmethod
    def _get_local_fidelity(probability_distribution: dict[int, float], num_qubits: int) -> float:
        """Process the probability distribution of a measurement to determine the
        local fidelity by averaging over single-qubit projectors.

        Args:
            probability_distribution: Obtained from the measurement result

        Returns:
            The local fidelity.
        """
        fidelity = 0.0
        for qubit in range(num_qubits):
            for bitstring, prob in probability_distribution.items():
                # Check whether the bit representing the current qubit is 0
                if not bitstring >> qubit & 1:
                    fidelity += prob / num_qubits
        return fidelity
