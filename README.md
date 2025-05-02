# Inference Time Improvement for MCTS

## Overview
This repository implements a method for improving inference time in the context of Monte Carlo Tree Search (MCTS) applied to language model token generation. The repository demonstrates two approaches:
- **MCTS-based token generation**: where tokens are generated using MCTS.
- **Standard token generation**: for comparison with the MCTS method.
  
You can use this to compare the performance between these two techniques.

## Environment Setup

This project uses the **uv package manager**. Follow the steps below to set up the environment:

1. Clone this repository:

2. Install the **uv package manager** (if you haven't already).

3. Create a virtual environment using uv:

   ```bash
   uv venv
   ```

4. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

5. Synchronize the environment:

   ```bash
   uv sync
   ```

## Example Usage

Once the virtual environment is set up and activated, you can run the example script to generate tokens using MCTS.

```bash
python3 scripts/main.py --n_tokens 20 --num_simulations 50 --select_k 5
```

* `n_tokens`: Number of tokens to generate.
* `num_simulations`: Number of rollouts (simulations) to perform during MCTS.
* `select_k`: Number of tokens to predict for the next word.

This script will show the token generation process using the specified MCTS parameters.

> **Note**: Currently, the input prompt is hardcoded into the `main.py` file. See the file for more details.

## Let's Have Fun!

Feel free to experiment with different settings and explore how the performance of MCTS-based token generation compares to standard methods.

