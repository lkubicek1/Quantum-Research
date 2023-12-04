import numpy as np
import random


class Gate:
    def __init__(self, matrix):
        self.matrix = matrix

    def apply(self, state, target_qubits, num_qubits):
        # If it's a single-qubit gate
        if self.matrix.shape == (2, 2):
            full_gate = 1
            for qubit in range(num_qubits):
                if qubit in target_qubits:
                    full_gate = np.kron(full_gate, self.matrix)
                else:
                    full_gate = np.kron(full_gate, np.eye(2))
            return np.dot(full_gate, state)

        # If it's a two-qubit gate like CNOT
        elif self.matrix.shape == (4, 4):
            control, target = target_qubits
            full_gate = 1
            for qubit in range(num_qubits):
                if qubit == control:
                    # Apply identity matrix for the control qubit
                    full_gate = np.kron(full_gate, np.eye(2))
                elif qubit == target:
                    # Apply CNOT matrix at the target qubit position
                    full_gate = np.kron(full_gate, self.matrix)
                else:
                    # Apply identity matrix for other qubits
                    full_gate = np.kron(full_gate, np.eye(2))

            # Resize the full_gate to fit the actual size of the quantum system
            full_gate = full_gate[:2 ** num_qubits, :2 ** num_qubits]

            return np.dot(full_gate, state)

        # If it's a three-qubit gate like Toffoli
        elif self.matrix.shape == (8, 8):
            control1, control2, target = target_qubits
            full_gate = 1
            for qubit in range(num_qubits):
                if qubit == control1 or qubit == control2:
                    full_gate = np.kron(full_gate, np.eye(2))  # Identity matrix for control qubits
                elif qubit == target:
                    full_gate = np.kron(full_gate, self.matrix)  # Toffoli matrix for the target qubit
                else:
                    full_gate = np.kron(full_gate, np.eye(2))  # Identity matrix for other qubits

            # Resize the full_gate to fit the actual size of the quantum system
            full_gate = full_gate[:2 ** num_qubits, :2 ** num_qubits]

            return np.dot(full_gate, state)


class PauliXGate(Gate):
    def __init__(self):
        super().__init__(np.array([[0, 1], [1, 0]], dtype=complex))


class PauliYGate(Gate):
    def __init__(self):
        super().__init__(np.array([[0, -1j], [1j, 0]], dtype=complex))


class PauliZGate(Gate):
    def __init__(self):
        super().__init__(np.array([[1, 0], [0, -1]], dtype=complex))


class HadamardGate(Gate):
    def __init__(self):
        super().__init__((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex))


class CNOTGate(Gate):
    def __init__(self):
        super().__init__(np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]], dtype=complex))


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


class QuantumRegister:
    # Define basis states |0⟩ and |1⟩ as constants
    STATE_0 = np.array([[1], [0]], dtype=complex)
    STATE_1 = np.array([[0], [1]], dtype=complex)

    def __init__(self, size=1, initial_state=None):
        self.size = size
        self.state = np.zeros((2 ** size, 1), dtype=complex)
        self.state[0, 0] = 1  # Initialize all qubits in the |0⟩ state

        # If an initial state is provided, use it
        if initial_state:
            self.set_state(initial_state)

    def set_state(self, new_state):
        if new_state.shape == (2 ** self.size, 1):
            self.state = new_state
        else:
            raise ValueError(f"new_state must be of shape {(2 ** self.size, 1)}")

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


    def __init__(self, num_qubits, noise_level = 0.0):
        self.num_qubits = num_qubits
        self.state = np.array([1] + [0] * (2 ** num_qubits - 1), dtype=complex).reshape(-1, 1)
        self.noise_level = noise_level 
        self.__in_noise__()

    def __in_noise__(self):
        for qubit in range(0, self.num_qubits):
            chance =random.randint(1, 100)
            if chance < self.noise_level*100:
                self.x([qubit])

    def __apply_gate__(self, gate, target_qubits):
        self.state = gate.apply(self.state, target_qubits, self.num_qubits)

    def x(self, target_qubits):
        self.__apply_gate__(self.__x__, target_qubits)

    def y(self, target_qubits):
        self.__apply_gate__(self.__y__, target_qubits)

    def z(self, target_qubits):
        self.__apply_gate__(self.__z__, target_qubits)

    def h(self, target_qubits):
        self.__apply_gate__(self.__h__, target_qubits)

    def cx(self, target_qubits):
        self.__apply_gate__(self.__cx__, target_qubits)

    def toffoli(self, target_qubits):
        self.__apply_gate__(self.__toffoli__, target_qubits)

    def measure(self):
        probabilities = np.abs(self.state.ravel()) ** 2
        result = np.random.choice(2 ** self.num_qubits, p=probabilities)
        self.state = np.zeros((2 ** self.num_qubits, 1), dtype=complex)
        self.state[result, 0] = 1
        return bin(result)[2:].zfill(self.num_qubits)
