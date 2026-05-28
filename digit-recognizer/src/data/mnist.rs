use std::fs::File;
use std::io::{self, BufReader, Read};

pub struct MnistData {
    pub images: Vec<Vec<f32>>, // each image: 784 values in [0.0, 1.0]
    pub labels: Vec<u8>,       // each label: digit 0-9
}

pub fn load_mnist(images_path: &str, labels_path: &str) -> io::Result<MnistData> {
    let images = load_images(images_path)?;
    let labels = load_labels(labels_path)?;
    assert_eq!(images.len(), labels.len(), "Image and label counts must match");
    Ok(MnistData { images, labels })
}

fn read_u32_be(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_images(path: &str) -> io::Result<Vec<Vec<f32>>> {
    let mut reader = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut reader)?;
    assert_eq!(magic, 2051, "Invalid MNIST image file magic number");

    let count = read_u32_be(&mut reader)? as usize;
    let rows  = read_u32_be(&mut reader)? as usize;
    let cols  = read_u32_be(&mut reader)? as usize;
    let pixels = rows * cols;

    let mut images = Vec::with_capacity(count);
    let mut buf = vec![0u8; pixels];

    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        let image = buf.iter().map(|&p| p as f32 / 255.0).collect();
        images.push(image);
    }

    Ok(images)
}

fn load_labels(path: &str) -> io::Result<Vec<u8>> {
    let mut reader = BufReader::new(File::open(path)?);

    let magic = read_u32_be(&mut reader)?;
    assert_eq!(magic, 2049, "Invalid MNIST label file magic number");

    let count = read_u32_be(&mut reader)? as usize;
    let mut labels = vec![0u8; count];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}
