use rand::prelude::*;
use std::{fmt::{Display, Formatter}, vec};
use crate::nn::utils::*;

pub struct Layer {
    // this structure represents a single layer of the network

    pub activations: Vec<f32>
}

impl Layer {
    pub fn new(size: usize) -> Layer {
        Layer {activations: vec![0.0; size]}
    }

    pub fn len(&self) -> usize {
        self.activations.len()
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}\n", self.activations)?;

        Ok(())
    }
}

pub struct Connection {
    // this structure represents the weights and biases between two layers of the network

    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>
}

impl Connection {
    pub fn new(prev: usize, next: usize) -> Connection {
        // create a new connection based on the sizes of previous and next layers
        // weights are initialized randomly, using Xavier weight initialization approach
        // biases are set to zero

        let mut rng = rand::rng();
        let mut weights = vec![vec![0.0; prev]; next]; 
        let limit = (6.0 / (prev + next) as f32).sqrt(); // Xavier weight initialization

        for i in 0..next {
            for j in 0..prev {
                weights[i][j] = rng.random_range(-limit..limit);
            }
        }

        let biases = vec![0.0; next];
        Connection { weights, biases }
    }
}

impl Display for Connection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Weights:\n")?;
        for i in 0..self.weights.len() {
            write!(f, " {:?}\n", self.weights[i])?;
        }

        write!(f, " Biases:\n")?;
        write!(f, " {:?}\n", self.biases)?;

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

        self.layers[0].activations = input;
    }

    pub fn feedforward(&mut self) {
        // fill in layers step by step based on first layer and connections

        for i in 0..self.connections.len() {
            let previous_layer = &self.layers[i].activations;
            let weights = &self.connections[i].weights;
            let biases =  &self.connections[i].biases;

            // multiply weight matrix by activation vector
            let weights_by_prev = matrix_by_vector(weights,previous_layer);

            // add biases vector
            let z = add_vectors(&weights_by_prev, biases);
            
            if i != self.connections.len() - 1 {
                for j in 0..z.len() {
                    // fill in calculated values to the next layer, after "sigmoid-ifying" them
                    self.layers[i + 1].activations[j] = sigmoid(z[j]);
                }
            }
            else {
                // if its last layer, softmax calculated values
                self.layers[i + 1].activations = softmax(&z)
            }
        }
    }

    pub fn backpropagation(&mut self, output: usize, lr: f32) {
        // gradient descent on connection parameters

        let n = self.layers.len();

        let mut logit_deriv: Vec<f32> = xentropy_grad(&self.layers[n - 1].activations, output);

        for i in (0..n - 1).rev() {
            let weights = &mut self.connections[i].weights;
            let previous_activations = &self.layers[i].activations;

            let weights_deriv = vec_by_vec_transposed(&logit_deriv, previous_activations);
            let biases_deriv= logit_deriv.clone();

            if i > 0 {
                logit_deriv = vec_mul(&matrix_by_vector(&transpose(weights), &logit_deriv), &sigmoid_derivative(previous_activations));
            }

            gradient_descent(weights, weights_deriv, lr);
            
            let biases = &mut self.connections[i].biases;
            gradient_descent_vector(biases, biases_deriv, lr);
        }
    }

    pub fn learn(&mut self, input: Vec<f32>, output: usize, lr: f32) {
        // learning step for neural network

        self.set_initial_layer(input);
        self.feedforward();
        self.backpropagation(output, lr);
    }

    pub fn predict(&mut self, input: Vec<f32>) -> u8 {
        // prediction of neural network

        self.set_initial_layer(input);
        self.feedforward();
        
        let output_layer = self.layers.last().unwrap();
        let mut label = 0_u8;
        let mut max = -1_f32;

        for i in 0..output_layer.activations.len() {
            let val = output_layer.activations[i];
            if val > max {
                label = i as u8;
                max = val;
            }
        }

        return label;
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