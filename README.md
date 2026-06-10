# Rust-Neural-Network

A neural network library for digit classification built in pure Rust. Educational project implementing backpropagation and gradient descent from scratch.

## What's Implemented

- **Activation Functions**: ReLU, Sigmoid
- **Classification**: Softmax
- **Training**: Backpropagation with mini-batch stochastic gradient descent
- **Weight Initialization**: Xavier initialization
- **MNIST Dataset**: Full training and evaluation pipeline

## Project Structure

```
digit-recognizer/     # Main neural network implementation
├── src/
│   ├── main.rs       # MNIST training and evaluation
│   ├── nn/           # Neural network core
│   │   ├── mlp.rs    # Multi-layer perceptron, layers, connections
│   │   └── math/     # Matrix operations and activation functions
│   └── data/
│       └── mnist.rs  # MNIST dataset loader
├── Cargo.toml
```

## Building

```bash
cd digit-recognizer
cargo build --release
```

## Running MNIST Training

```bash
cargo run --release
```

The network trains on MNIST images (28x28 → 784 input neurons) with configurable hidden layers and outputs 10 classes (digits 0-9).

## Example Network

The default configuration:
- Input layer: 784 neurons (28×28 pixels)
- Hidden layers: 32 neurons (ReLU), 32 neurons (ReLU)
- Output layer: 10 neurons (Softmax for classification)
- Training: 10 epochs, learning rate 0.005, batch size 10

## License

MIT
