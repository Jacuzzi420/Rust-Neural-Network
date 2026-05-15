mod nn;
mod data;
use std::time::Instant;
fn main() {
    let training_data = data::mnist::load_mnist(
        "mnist/train-images.idx3-ubyte",
        "mnist/train-labels.idx1-ubyte",
    ).expect("Failed to load MNIST training_data");

    let testing_data = data::mnist::load_mnist(
        "mnist/t10k-images.idx3-ubyte",
        "mnist/t10k-labels.idx1-ubyte",
    ).expect("Failed to load MNIST training_data");

    println!("Loaded {} training images", training_data.images.len());
    println!("First image label: {}", training_data.labels[0]);

    let mut network = nn::mlp::Network::new(vec![784, 16, 16, 10]);

    let epochs = 20;

    for e in 0..epochs {
        println!("Epoch {} started", e);
        let now = Instant::now();

        for i in 0..training_data.images.len() {
            network.learn(training_data.images[i].clone(), training_data.labels[i] as usize, 0.1);
        }


        let elapsed_time = now.elapsed();
        println!("Epoch {} done in {} seconds", e, elapsed_time.as_secs());

        let mut correct = 0;
        let total = testing_data.images.len();
        for i in 0..total {
            let prediction = network.predict(testing_data.images[i].clone());

            if prediction == testing_data.labels[i] {
                correct += 1;
            }
        }
        println!("Correct: {} Total: {} Accuracy: {}", correct, total, correct as f32 / total as f32);
    }
}


