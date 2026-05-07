/*
Pomysł:
1. Struktura opisująca sieć - budowa
Struktura network w inicie przyjmuje tablicę, np. [784, 16, 16, 10] (size = 4)
każdy element reprezentuje osobny layer

atrybuty struktury:
weights[size - 1]
weights[i - 1] przechowuje macierz wag pomiędzy layer[i - 1] a layer[i]
weights[0] to macierz 784x16
weights[1] 16x16
weights[2] 16x10

biases[size - 1]
biases[i - 1] przechowuje wektor biasów layer[i]


elementy biases i weights początkowo wypełniamy losowo

wszystko powyższe dajemt do inita


2. Funkcja feed forward:
input -> output
input czyli wypełniony pierwszy layer
na podstawie pierwszego layera obliczamy następne aż do outputu (mamy już wagi i biasy)

Potrzebny będzie sigmoid 


3. Spróbować wczytać MNIST i wyciągnąć pierwszy layer (może być vibecoded)


4. (ewentualnie) Prosta implementacja trenowania
*/


use rand::prelude::*;

struct Layer {
    neurons: Vec<f32>
}

impl Layer {
    fn new(size: usize) -> Layer {
        Layer {neurons: vec![0.0; size]}
    }
}

struct Connection {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>
}

impl Connection {
    fn new(prev: usize, next: usize) -> Connection {
        let mut rng = rand::rng();
        let weights = vec![vec![rng.random_range(-1.0..1.0); prev]; next]; // TODO: choose a better way of selecting initial weights
        let biases = vec![0.0; next];
        Connection {weights: weights, biases: biases}
    }
}

struct Network {
    layers: Vec<Layer>,
    connections: Vec<Connection>
}

impl Network {
    fn new(sizes: Vec<usize>) -> Network {
        let mut layers = Vec::new();
        
        for i in 0..sizes.len() {
            let layer = Layer::new(sizes[i]);
            layers.push(layer);
        }

        let mut connections = Vec::new();

        for i in 0..sizes.len()-1 {
            let connection = Connection::new(sizes[i], sizes[i + 1]);
            connections.push(connection);
        }

        Network {layers: layers, connections: connections}
    }  
}


fn main() {
    println!("Hello, world!");
}
