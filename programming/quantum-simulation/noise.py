from qsim import *

qc = QuantumCircuit(num_qubits= 2, noise_level=.9)
print (qc.measure())
