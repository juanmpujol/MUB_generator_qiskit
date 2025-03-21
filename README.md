# MUB Generator with Qiskit

This script generates the full set of mutually unbiased bases (MUBs) for a given number of qubits between 2 and 10 using Qiskit.

Adapted from the article ["An Efficient Quantum Circuit Construction Method for Mutually Unbiased Bases in $n$-Qubit Systems", Wang Yu, Wu Dongsheng](https://arxiv.org/abs/2311.11698)


## Requirements

- Python 3.x
- Qiskit
- NumPy
- tqdm

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/MUB_generator_qiskit.git
    cd MUB_generator_qiskit
    ```

2. Install the required Python packages:
    ```sh
    pip install qiskit numpy tqdm
    ```

## Usage

Run the script and follow the prompt to input the number of qubits (1 to 10):
```sh
python MUB_generator_qiskit.py
```

The complete set of bases is then saved in the current directory as both a .npy file and .txt file
