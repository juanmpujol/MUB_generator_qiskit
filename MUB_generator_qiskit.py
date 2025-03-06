
import numpy as np
from qiskit import QuantumCircuit, quantum_info
from itertools import combinations
from tqdm import tqdm  # Add this import


## The first step is to generate an irreducible polynomial.
# For N qubits, we will need a polynomial of degree N.
# The polynomial is represented as a list of coefficients, where the ith element is the coefficient of x^i, up to x^(N-1).
# The arrays corresponding to polynomials up to the 10th degree are loaded in the dictionary irr_pols.

irr_pols = {
    1: [],
    2: [1],
    3: [1, 0],
    4: [1, 0, 0],
    5: [0, 1, 0, 0],
    6: [1, 0, 0, 0, 0],
    7: [1, 0, 0, 0, 0, 0],
    8: [1, 0, 1, 1, 0, 0, 0],
    9: [1, 0, 0, 0, 0, 0, 0, 0],
    10: [0, 0, 1, 0, 0, 0, 0, 0, 0],
}

## nq = number of qubits
# The user is asked to input the number of qubits, in order to select the corresponding polynomial.

nq = int(input("How many qubits (1 to 10)? "))
if nq < 1 or nq > 10:
    raise ValueError("Invalid number of qubits")


## Here I define the auxiliary functions integer_form and vector_form
# integer_form takes a list of 0s and 1s and returns the integer it represents
# vector_form takes an integer and returns the list of 0s and 1s it represents
# These are necessary for the calculations that follow

def integer_form(x):
    return sum([x[i]*2**i for i in range(len(x)) ])

def vector_form(x):
    binary_list = [int(i) for i in bin(x)[2:]]
    binary_list.reverse()
    auxzeros = [0]*(nq-len(binary_list))
    binary_list.extend(auxzeros)
    return binary_list


## We now define the matrices M_0 and M_1 
# For this, we need to calculate the vectorial forms of the x^m, with m = 0, ..., 2n-2
# We then apply ec. 2 to obtain the matrices M_0 and M_1


x = np.zeros((2*nq-1, nq))
id = np.eye(nq)
for i in range(nq):
    x[i] = id[i, :]

x[nq] = np.concatenate(([1], irr_pols[nq]))

for i in range(nq+1, 2*nq-1):
	x[i] =  np.concatenate(([0], x[i-1][:-1])) + x[i-1][-1] * x[nq]

rows_0 = np.zeros((nq, nq))
for i in range(nq):
	rows_0[i] = [x[j+i][0] for j in range(nq)]

rows_1 = np.zeros((nq, nq))
for i in range(nq):
	rows_1[i] = [x[j+i][1] for j in range(nq)]


M_0 = np.vstack(rows_0)
M_1 = np.vstack(rows_1)


## Acá podría calcular los vectores M_0(x^m)^t y M_1(x^m)^t 
# Esto sería para que sea más rápido al correr el cálculo de los a(j, r), pero por ahora no es necesario


## Now we define the functions to obtain the coefficients a_r(j) and b_{s+t}(j)
# The products are evaluated modulo 2


def a(j, r):
    a1 = (vector_form(j) @ (M_0 @ x[2*r].T)) % 2
    a2 = (vector_form(j) @ (M_1 @ x[2*r].T)) % 2
    if a1 == 0:
        return 0 if a2 == 0 else (2 if a2 == 1 else None)
    elif a1 == 1:
        return 3 if a2 == 0 else (1 if a2 == 1 else None)
    else:
        return None

def b(j, s, t):
	return (vector_form(j)@(M_0 @ x[s+t].T)) % 2

## Now we define the function that generates the states of the basis
# The function takes the number of qubits nq, the index j of the basis, and the index of the state in the basis
# The function is implemented in qiskit and returns the state as a vector


def get_state(nq, j, state_index):
    
    qc = QuantumCircuit(nq)

    ## This is to prepare the initial states
    # We need to apply the U(j) circuit to all the elements of the canonical basis
    # This is the simplest way I found to prepare the states of the canonical basis, applying x gates to the qubits to form all possible 0-1 combinations

    combs = np.array([set()] + [combo for i in range(1, nq + 1) for combo in combinations(range(nq), i)], dtype = object)
    relevant_combs = combs[state_index]
    for i in range(len(relevant_combs)):
        qc.x(relevant_combs[i])

    ## First we apply a Haddamard gate to all qubits
    ## Then we apply the corresponding S gates according to the coefficients a(j, s)

    for qubit in range(nq):
        qc.h(qubit)
        for i in range(a(j, qubit)):
            qc.s(qubit)
    
    ## Now we apply the CZ gates according to the coefficients b(s, t)

    for s in range(nq):
        for t in range(s+1, nq):
            if b(j, s, t) == 1: 
                qc.cz(s,t)

    sv = quantum_info.Statevector.from_instruction(qc).data
    return sv


## This is the function to obtain all states of the jth basis of the full set for nq qubits
# The function returns a matrix where each row is a state of the basis

def full_basis(nq, j):
    all_states = []
    for i in range(2**nq):
        all_states.append(get_state(nq, j, i))
    #print("Basis {} done\n".format(j))
    return np.array(all_states)
    
## This function generates the full set of bases for nq qubits,
# The function returns a 3D array, where the first index is the basis the second index is the state, and the third index is the amplitude of the state

def full_set(nq):
    set_matrix = []
    set_matrix.append(np.eye(2**nq))
    for j in tqdm(range(2**nq), desc="Generating full set"):  # Wrap loop with tqdm
        set_matrix.append(full_basis(nq, j))
    return np.array(set_matrix)
        
fullset_array = full_set(nq)    

## Now we save the bases in a txt file and in a numpy file and a txt file


with open('full_mub_' + str(nq) + 'qbits.txt', 'w') as f:
    for j in range(2**nq):
        f.write("\n Basis nr " + str(j) + " :\n")
        np.savetxt(f, fullset_array[j], fmt = '%.3f', delimiter = "," )
        
np.save("full_mubs_" + str(nq) + "qbits", fullset_array) 

print("Done!")
print(".npy and .txt files saved in current directory")

