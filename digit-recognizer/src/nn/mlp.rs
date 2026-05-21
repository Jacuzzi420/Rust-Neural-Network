use std::{fmt::{Display, Formatter}};
use crate::nn::math::matrix::*;
use crate::nn::math::utils::*;

pub struct Layer {
    // this structure represents a single layer of the network

    pub activations: Matrix,
}

impl Layer {
    pub fn new(size: usize) -> Layer {
        Layer {activations: Matrix::zeros((size, 1))}
    }

    pub fn len(&self) -> usize {
        self.activations.shape.0
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n", self.activations)?;

        Ok(())
    }
}

pub struct Connection {
    // this structure represents the weights and biases between two layers of the network

    pub weights: Matrix,
    pub biases: Matrix,
}

impl Connection {
    pub fn new(prev: usize, next: usize) -> Connection {
        // create a new connection based on the sizes of previous and next layers
        // weights are initialized randomly, using Xavier weight initialization approach
        // biases are set to zero

        let limit = (6.0 / (prev + next) as f32).sqrt(); // Xavier weight initialization
        let weights = Matrix::rand((next, prev), -limit, limit);
        let biases = Matrix::zeros((next, 1));
        Connection { weights, biases }
    }
}

impl Display for Connection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weights:\n")?;
        write!(f, "{}\n", self.weights)?;

        write!(f, " Biases:\n")?;
        write!(f, " {}\n", self.biases)?;

        Ok(())
    }
}

pub struct Network {
    // this structure represents the whole neural network
    // consisting of N layers and N - 1 connections

    pub layers: Vec<Layer>,
    pub connections: Vec<Connection>
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Network {
        // create new neural network
        // for example, Network::new([4, 3, 3, 2]) will create a network with four layers
        // 1st layer will have 4 neurons, 2nd - 3, 3rd - 3, 4th - 2
        // weights are initialized randomly, biases are set to 0

        let mut layers = Vec::new();
        
        for i in 0..sizes.len() {
            // create sizes.len() layers
            let layer = Layer::new(sizes[i]);
            layers.push(layer);
        }

        let mut connections = Vec::new();

        for i in 0..sizes.len()-1 {
            // create sizes.len() - 1 connections
            let connection = Connection::new(sizes[i], sizes[i + 1]);
            connections.push(connection);
        }

        Network { layers, connections }
    }

    pub fn set_initial_layer(&mut self, input: Vec<f32>) {
        // set the input (first layer) of the network
        assert_eq!(input.len(), self.layers[0].len(), "Wrong initial layer size!");
        
        for i in 0..input.len() {
            assert!(0.0 <= input[i] && input[i] <= 1.0, "Activation values must be in [0, 1] range!");
        }

        self.layers[0].activations = Matrix::new((input.len(), 1), input);
    }

    pub fn feedforward(&mut self) {
        // fill in layers step by step based on first layer and connections

        for i in 0..self.connections.len() {
            let previous_layer = &self.layers[i].activations;
            let weights = &self.connections[i].weights;
            let biases =  &self.connections[i].biases;

            // multiply weight matrix by activation vector
            let weights_by_prev = weights.mat_mul(previous_layer);

            // add biases vector
            let z = weights_by_prev.add(biases);
            
            if i != self.connections.len() - 1 {
                // fill in calculated values to the next layer, after "sigmoid-ifying" them
                self.layers[i + 1].activations = mat_sigmoid(z);
            }
            else {
                // if its last layer, softmax calculated values
                self.layers[i + 1].activations = mat_softmax(z);
            }
        }
    }

    pub fn backpropagation(&mut self, output: usize) -> (Vec<Matrix>, Vec<Matrix>) {
        // backpropagation step for neural network

        let n = self.layers.len();
        let activations = &self.layers[n - 1].activations;

        let mut layer_grad = xentropy_grad(activations, output);

        let mut weights_grads: Vec<Matrix> = Vec::new();
        let mut biases_grads: Vec<Matrix> = Vec::new();

        for i in (0..n - 1).rev() {
            let weights = &mut self.connections[i].weights;
            let previous_activations = &self.layers[i].activations;

            let weights_deriv = &layer_grad.mat_mul(&previous_activations.transpose());
            let biases_deriv = layer_grad.clone();
            
            if i > 0 {
                layer_grad = weights.transpose().mat_mul(&layer_grad).mul(&sigmoid_grad(previous_activations));
            }

            weights_grads.push(weights_deriv.clone());
            biases_grads.push(biases_deriv.clone());
        }

        weights_grads.reverse();
        biases_grads.reverse();

        (weights_grads, biases_grads)
    }

    pub fn get_gradient(&mut self, input: Vec<f32>, output: usize) -> (Vec<Matrix>, Vec<Matrix>) {
        // calculates gradient for given input and output

        self.set_initial_layer(input);
        self.feedforward();
        self.backpropagation(output)
    }

    pub fn learn(&mut self, input: Vec<f32>, output: usize, lr: f32) {
        let (weights_grads, biases_grads) = self.get_gradient(input.clone(), output);
    
        for i in 0..self.connections.len() {
            self.connections[i].weights = gradient_descent(&self.connections[i].weights, &weights_grads[i], lr);
            self.connections[i].biases = gradient_descent(&self.connections[i].biases, &biases_grads[i], lr);
        }
    }

    pub fn learn_on_batch(&mut self, inputs: Vec<Vec<f32>>, outputs: Vec<usize>, lr: f32) {
        let n = inputs.len();

        let mut batch_weights_grads =Vec::<Matrix>::new();
        let mut batch_biases_grads = Vec::<Matrix>::new();

        for i in 0..n {
            let (weights_grads, biases_grads) = self.get_gradient(inputs[i].clone(), outputs[i]);

            if i == 0 {
                batch_weights_grads = weights_grads.clone();
                batch_biases_grads = biases_grads.clone();
            } 
            else {
                for l in 0..weights_grads.len() {
                    batch_weights_grads[l] = batch_weights_grads[l].add(&weights_grads[l]);
                    batch_biases_grads[l] = batch_biases_grads[l].add(&biases_grads[l]);
                }
            }
        }

        let avg_lr = lr / n as f32;
        for l in 0..self.connections.len() {
            self.connections[l].weights = gradient_descent(&self.connections[l].weights, &batch_weights_grads[l], avg_lr);
            self.connections[l].biases = gradient_descent(&self.connections[l].biases, &batch_biases_grads[l], avg_lr);
        }
    }

    pub fn predict(&mut self, input: Vec<f32>) -> u8 {
        // prediction of neural network

        self.set_initial_layer(input);
        self.feedforward();
        
        let output_layer = self.layers.last().unwrap();
        let mut label = 0;
        let mut max = -1.;

        for i in 0..output_layer.activations.shape.0 {
            let val = output_layer.activations[(i, 0)];
            if val > max {
                label = i;
                max = val;
            }
        }

        label as u8
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let last = self.connections.len();

        for i in 0..last {
            writeln!(f, "Layer {}:\n {}", i, self.layers[i])?;
            writeln!(f, "Connection {}:\n {}", i, self.connections[i])?;
        }

        writeln!(f, "Layer {}:\n {}", last, self.layers[last])?;

        Ok(())
    }
}