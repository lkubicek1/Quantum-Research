import numpy as np

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
    def __init__(self, *quantum_registers):
        self.quantum_registers = quantum_registers
        self.size = sum([qr.size for qr in quantum_registers])

    def get_registers(self):
        return self.quantum_registers

    def apply_pauli_x(self, register_index):
        # Apply the Pauli-X gate
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        state = self.quantum_registers[register_index].get_state()
        new_state = np.dot(pauli_x, state)

        # Update the state of the quantum register
        self.quantum_registers[register_index].set_state(new_state)
