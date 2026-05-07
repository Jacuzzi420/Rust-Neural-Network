mod mnist;

use rand::prelude::*;
use std::fmt::{Display, Formatter};

struct Layer {
    // this structure represents a single layer of the network

    activations: Vec<f32>
}

impl Layer {
    fn new(size: usize) -> Layer {
        Layer {activations: vec![0.0; size]}
    }

    fn len(&self) -> usize {
        self.activations.len()
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}\n", self.activations)?;

        Ok(())
    }
}

struct Connection {
    // this structure represents the weights and biases between two layers of the network

    weights: Vec<Vec<f32>>,
    biases: Vec<f32>
}

impl Connection {
    fn new(prev: usize, next: usize) -> Connection {
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

struct Network {
    // this structure represents the whole neural network
    // consisting of N layers and N - 1 connections

    layers: Vec<Layer>,
    connections: Vec<Connection>
}

impl Network {
    fn new(sizes: Vec<usize>) -> Network {
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

    fn set_initial_layer(&mut self, input: Vec<f32>) {
        // set the input (first layer) of the network
        assert_eq!(input.len(), self.layers[0].len(), "Wrong initial layer size!");
        
        for i in 0..input.len() {
            assert!(0.0 <= input[i] && input[i] <= 1.0, "Activation values must be in [0, 1] range!");
        }

        self.layers[0].activations = input;
    }

    fn feedforward(&mut self) {
        // fill in layers step by step based on first layer and connections

        for i in 0..self.connections.len() {
            let previous_layer = &self.layers[i].activations;
            let weights = &self.connections[i].weights;
            let biases =  &self.connections[i].biases;

            // multiply weight matrix by activation vector
            let weights_by_prev = matrix_by_vector(weights,previous_layer);

            // add biases vector
            let weights_by_prev_plus_biases = add_vectors(&weights_by_prev, biases);

            assert_eq!(weights_by_prev_plus_biases.len(), self.layers[i + 1].len(), "Wrong calculated layer size!");

            for j in 0..self.layers[i + 1].len() {
                // fill in calculated values to the next layer, after "sigmoid-ifying" them
                self.layers[i + 1].activations[j] = sigmoid(weights_by_prev_plus_biases[j])
            }
        }
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

fn sigmoid(x: f32) -> f32 {
    // sigmoid function to "squish" values into the [0, 1] range
    1.0 / (1.0 + (-x).exp())
}

fn matrix_by_vector(m: &Vec<Vec<f32>>, v: &Vec<f32>) -> Vec<f32> {
    // multiply matrix M by vector V
    // M: n x p; V: p x 1
    // n - next, p - prev

    let p = m[0].len();
    let n = m.len();

    assert_eq!(p, v.len(), "Wrong matrix/vector size!");

    let mut result = Vec::<f32>::new();

    for i in 0..n {
        let mut element = 0.0;

        for j in 0..p {
            element += m[i][j] * v[j];
        }

        result.push(element);
    }

    result
}

fn add_vectors(u: &Vec<f32>, v: &Vec<f32>) -> Vec<f32> {
    // add vectors U and V

    let n = u.len();

    assert_eq!(n, v.len(), "Vectors must have the same length!");

    let mut result = Vec::<f32>::new();

    for i in 0..n {
        result.push(u[i] + v[i]);
    }
    
    result
}

fn main() {
    // example network:
    let mut exmpl_network = Network::new(vec![4, 3, 3, 2]); // 4 -> 3 -> 3 -> 2 network
    exmpl_network.set_initial_layer(vec![0.6, 0.0, 1.0, 1.0]);

    println!("\n\nINITIAL NETWORK:\n\n{}", exmpl_network);

    exmpl_network.feedforward();

    println!("\n\nNETWORK AFTER FEED FORWARD:\n\n{}", exmpl_network);


    let data = mnist::load_mnist(
        "mnist/train-images.idx3-ubyte",
        "mnist/train-labels.idx1-ubyte",
    ).expect("Failed to load MNIST data");

    println!("Loaded {} training images", data.images.len());
    println!("First image label: {}", data.labels[0]);

    let mut network = Network::new(vec![784, 16, 16, 10]);
    network.set_initial_layer(data.images[0].clone());
    network.feedforward();

    println!("Output layer: {:?}", network.layers.last().unwrap().activations);
}


#[cfg(test)]
mod tests {
    use super::*;
 
    #[test]
    fn test_matrix_by_vector() {
        // test matrix_by_vector function

        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let v = vec![3.0, 2.0, 1.0];
        assert_eq!(matrix_by_vector(&m, &v), vec![10.0, 28.0]);
    }

    #[test]
    fn test_add_vectors() {
        // test add_vectors function

        let u = vec![-0.5, 2.6, 4.0];
        let v = vec![0.0, 0.4, -5.0];
        assert_eq!(add_vectors(&u, &v), vec![-0.5, 3.0, -1.0]);
    }
}