# GPU-CPU-Stress-Test

This repository provides a Python-based stress test for GPUs and CPUs. 
The test generates random data and trains a Variational Autoencoder (VAE) model, measuring system performance in terms of memory usage, training time, and hardware utilization. 
This can be useful for evaluating hardware capabilities and debugging performance bottlenecks.

## Usage
To run the test, execute:
`python stress_test.py --device cuda #for GPU`
or 
`python stress_test.py --device cpu   # For CPU`

Alternatively, use the provided bash script:
`bash run_test.sh`

## Customization
Modify the script parameters to test different configurations:
`python stress_test.py --device cuda --batch_size 512 --epochs 20`

# Licence
This project is open-source and available under the MIT License.

# Contributions
Feel free to submit issues or pull requests to improve the test.
