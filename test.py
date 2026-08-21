import pennylane as qml
from catalyst import qjit

dev = qml.device("lightning.qubit", wires=2)

@qjit
@qml.qnode(dev)
def circuit():
    return qml.expval(qml.PauliZ(0))

print(circuit())