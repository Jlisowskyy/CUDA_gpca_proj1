# MCTS-Checkmate-Chariot

A CUDA-accelerated Monte Carlo Tree Search (MCTS) chess engine designed to maximize computational efficiency by leveraging both the CPU and GPU. This project explores various implementations of board simulations on CUDA and parallel MCTS traversal on the CPU.

## Features
- **Hybrid CPU-GPU Design**: 
  - Tree traversal is performed on the CPU.
  - Game simulations are executed on the GPU.
- **Two CUDA Implementations for Board Simulations**:
  1. **1-thread-per-board**: Each thread simulates an independent chess position.
  2. **6-threads-per-board (split by piece type)**: Each board is evaluated using multiple threads assigned to different piece types.
- **Parallel CPU MCTS Traversal**:
  - Utilizes multi-threading to maximize CPU performance.
  - Ensures efficient resource utilization between CPU and GPU.
- **Validated Move Generation**:
  - Results were compared against the prior chess engine, [Checkmate-Chariot](https://github.com/Jlisowskyy/Checkmate-Chariot), and yielded identical outcomes.

## Installation
### Prerequisites
- CUDA-enabled GPU
- NVIDIA CUDA Toolkit
- C++ compiler with OpenMP support
- CMake (for build automation)

### Build Instructions
```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j$(nproc)
```

## Usage

Start the application, engine comes together with simplistic TUI which is easy to navigate and well explained.

## Performance Evaluation
- Benchmarked across various CPU and GPU configurations.
- Verified move generation correctness through cross-validation with Checkmate-Chariot.

## Future Improvements
- Enhancing parallelization efficiency for MCTS backpropagation.
- Experimenting with different CUDA kernel optimizations.
- Extending support for different evaluation heuristics.

## License
This project is licensed under the MIT License.
