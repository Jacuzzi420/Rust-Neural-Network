mod mlp;
mod mnist;
mod tests;

fn main() {
    // example network:
    let mut exmpl_network = mlp::Network::new(vec![4, 3, 3, 2]); // 4 -> 3 -> 3 -> 2 network
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

    let mut network = mlp::Network::new(vec![784, 16, 16, 10]);
    network.set_initial_layer(data.images[0].clone());
    network.feedforward();

    println!("Output layer: {:?}", network.layers.last().unwrap().activations);
}


