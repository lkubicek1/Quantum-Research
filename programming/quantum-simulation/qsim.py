import random

import numpy as np


class Gate:
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, state, target_qubits, num_qubits):
        full_gate = 1
        for qubit in range(num_qubits):
            if qubit in target_qubits:
                full_gate = np.kron(full_gate, self.matrix)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        return full_gate


class PauliXGate(Gate):
    def __init__(self):
        super().__init__(np.array([[0, 1], [1, 0]], dtype=complex))

    def apply(self, state, target_qubits, num_qubits):
        full_gate = super().apply(state, target_qubits, num_qubits)
        return np.dot(full_gate, state)


class PauliYGate(Gate):
    def __init__(self):
        super().__init__(np.array([[0, -1j], [1j, 0]], dtype=complex))

    def apply(self, state, target_qubits, num_qubits):
        full_gate = super().apply(state, target_qubits, num_qubits)
        return np.dot(full_gate, state)


class PauliZGate(Gate):
    def __init__(self):
        super().__init__(np.array([[1, 0], [0, -1]], dtype=complex))

    def apply(self, state, target_qubits, num_qubits):
        full_gate = super().apply(state, target_qubits, num_qubits)
        return np.dot(full_gate, state)


class HadamardGate(Gate):
    def __init__(self):
        super().__init__((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))

    def apply(self, state, target_qubits, num_qubits):
        full_gate = super().apply(state, target_qubits, num_qubits)
        return np.dot(full_gate, state)


class CNOTGate(Gate):
    def __init__(self):
        super().__init__(np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]], dtype=complex))

    def apply(self, state, target_qubits, num_qubits):
        control, target = target_qubits
        full_gate = super().apply(state, [target], num_qubits)

        # Resize the full_gate to fit the actual size of the quantum system
        full_gate = full_gate[:2 ** num_qubits, :2 ** num_qubits]

        return np.dot(full_gate, state)


class ToffoliGate(Gate):
    def __init__(self):
        super().__init__(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 1],
                                   [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex))

    def apply(self, state, target_qubits, num_qubits):
        control1, control2, target = target_qubits
        full_gate = super().apply(state, [target], num_qubits)

        # Resize the full_gate to fit the actual size of the quantum system
        full_gate = full_gate[:2 ** num_qubits, :2 ** num_qubits]

        return np.dot(full_gate, state)


class QuantumRegister:

    def __init__(self, size=1):
        self.num_qubits = size
        self.state = np.array([1] + [0] * (2 ** size - 1), dtype=complex).reshape(-1, 1)

    def __apply_gate__(self, gate, target_qubits):
        self.state = gate.apply(self.state, target_qubits, self.num_qubits)

    def get_state(self):
        """Return the current state of the quantum register."""
        return self.state


class QuantumCircuit:
    __x__ = PauliXGate()
    __y__ = PauliYGate()
    __z__ = PauliZGate()
    __h__ = HadamardGate()
    __cx__ = CNOTGate()
    __toffoli__ = ToffoliGate()

    def __init__(self, num_qubits, noise_level=0.0):
        self.quantum_register = QuantumRegister(size=num_qubits)
        self.noise_level = noise_level
        self.__add_noise__()

    def __add_noise__(self):
        for qubit in range(0, self.quantum_register.num_qubits):
            chance = random.randint(1, 100)
            if chance < self.noise_level * 100:
                self.x([qubit])

    def measure(self):
        probabilities = np.abs(self.quantum_register.get_state().ravel()) ** 2
        result = np.random.choice(2 ** self.quantum_register.num_qubits, p=probabilities)
        self.quantum_register.state = np.zeros((2 ** self.quantum_register.num_qubits, 1), dtype=complex)
        self.quantum_register.state[result, 0] = 1
        return bin(result)[2:].zfill(self.quantum_register.num_qubits)

    def x(self, target_qubits):
        self.quantum_register.__apply_gate__(self.__x__, target_qubits)

    def y(self, target_qubits):
        self.quantum_register.__apply_gate__(self.__y__, target_qubits)

    def z(self, target_qubits):
        self.quantum_register.__apply_gate__(self.__z__, target_qubits)

    def h(self, target_qubits):
        self.quantum_register.__apply_gate__(self.__h__, target_qubits)

    def cx(self, target_qubits):
        self.quantum_register.__apply_gate__(self.__cx__, target_qubits)

    def toffoli(self, target_qubits):
        self.quantum_register.__apply_gate__(self.__toffoli__, target_qubits)
