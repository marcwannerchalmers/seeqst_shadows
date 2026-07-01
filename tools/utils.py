
from itertools import chain
from typing import Any
import pennylane as qp
import numpy as np
from abc import ABC, abstractmethod
from pennylane.operation import Operator

# Credit goes to SEEQST
# Can probably be tuned
# TODO: remove gate info
def build_parallel_entangler_blocks(selective_block, num_qubits, xy_ind: int):
    """
    Build parallel GHZ-style entangling gate sequences for A SINGLE selective block.
    """
    block = selective_block
    # encodes block as binary
    bin_str = format(block, f'0{num_qubits}b')
    # can be improved via bitwise operations
    active_qubits = [i for i, bit in enumerate(bin_str[::1]) if bit == '1']  # LSB = qubit 0
    # print(bin_str, active_qubits)
    if not active_qubits: # here, one simply seems to measure computational basis
        return '' # No gates needed

    sequence = []

    # Step 1: Initial rotation on first qubit (arbitrary choice)
    XY = "X" if xy_ind == 0 else "Y"
    sequence.append(str(f'(R{XY}90:{active_qubits[0]})'))

    head = [active_qubits[0]]
    tail = active_qubits[1:]

    # Step 2: GHZ layering: use ALL head qubits as controls
    while tail:
        new_tail = []
        for h in head:
            if not tail:
                break
            # Assign one tail target to this control
            tgt = tail.pop(0)
            sequence.append(str(f'(CNOT:{h},{tgt})'))
            new_tail.append(tgt)
        head.extend(new_tail)

    # Step 3: Create RX90 version of same circuit
    #rx_sequence = [gate.replace('RY90', 'RX90') for gate in sequence]

    # Step 4: Return both sequences in reverse order
    #all_sequences.append([''.join(sequence[::-1]), ''.join(rx_sequence[::-1])])

    return sequence

# change this to Pennylane

def parse_circuit(circuit_text, initial_text=""):
    """
    Convert a text-based circuit description into Qiskit circuits.
    
    Args:
        text_circuits (list of str): List of circuit descriptions in text format.
        n_qubits (int): Number of qubits and classical bits in each circuit.

    Returns:
        list of QuantumCircuit: List of Qiskit QuantumCircuit objects.
    """
    gates = []

    circuit_text=initial_text+circuit_text

    # Split operations while handling concatenated gates
    operations = circuit_text.split(')')

    for op in operations:
        if not op.strip():  # Skip empty parts
            continue
        op = op.strip().strip('(')  # Remove leading (
        gate_info = op.split(':')

        if len(gate_info) < 2:
            continue  # Skip invalid formats

        gate_name = gate_info[0]
        qubit_indices = list(map(int, gate_info[1].split(',')))  # Extract qubit indices

        # Map gate names to Qiskit gates
        if gate_name == "RX90":
            gates.append(qp.RX(np.pi/2, wires=qubit_indices[0]))
        elif gate_name == "RY90":
            gates.append(qp.RY(np.pi/2, wires=qubit_indices[0]))
        elif gate_name == "CNOT":
            gates.append(qp.CNOT(wires=[qubit_indices[0], qubit_indices[1]]))
        elif gate_name == "H":
            gates.append(qp.Hadamard(wires=qubit_indices[0]))
        elif gate_name == "MEAS":
            raise ValueError(f"Unsupported gate: {gate_name}")
        else:
            raise ValueError(f"Unsupported gate: {gate_name}")

    return gates

def flatten_list(nested_list):
    """Flattens a list of lists into a single list using itertools.chain."""
    return list(chain(*nested_list))

def post_meas_state_gates(outcome, one_ev=-1):
    for i, oc in enumerate(outcome):
        if oc == one_ev:
            qp.X(i)

