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

    let n = training_data.images.len();
    println!("Loaded {} training images", n);
    println!("First image label: {}", training_data.labels[0]);

    let mut network = nn::mlp::Network::new(vec![784, 32, 32, 10]);

    let epochs = 10;
    let lr = 3.0;
    let batch_size = 10;

    for e in 0..epochs {
        println!("Epoch {} started", e);
        let now = Instant::now();
        let shuffled = nn::utils::shuffle(n);
        let mut k = 0;

        while k + batch_size <= n {
            // new iteration - new batch
            let mut batch_images = Vec::new();
            let mut batch_labels = Vec::new();
            
            for i in k..k+batch_size {
                let idx = shuffled[i];
                batch_images.push(training_data.images[idx].clone());
                batch_labels.push(training_data.labels[idx] as usize);
            }

            network.learn_on_batch(batch_images, batch_labels, lr);
            k += batch_size;
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


