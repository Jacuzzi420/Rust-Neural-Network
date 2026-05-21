use crate::nn::math::matrix::*;
use rand::prelude::*;

// TODO: Dodaj relu i sigmoid na output
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_grad(s: &Matrix) -> Matrix {
    // returns matrix of sigmoid derivative from sigmoid matrix
    // s - sigmoid matrix

    let mut result = Matrix::zeros(s.shape);

    for i in 0..result.shape.0 {
        for j in 0..result.shape.1 {
            result[(i, j)] = s[(i, j)] * (1. - s[(i, j)]);
        }
    }

    result
}

pub fn xentropy(s: &Matrix, t: usize) -> f32 {
    // cross entropy - cost function for classification
    // s - vector with probability distribution, t - class label

    assert_eq!(s.shape.1, 1, "Matrix has to be column vector");
    assert!(t < s.shape.0, "Label too big");

    return -(s[(t, 0)].clamp(10e-9_f32, 1_f32)).ln()
}

pub fn xentropy_grad(s: &Matrix, t: usize) -> Matrix {
    // cross entropy gradient
    // s - vector with probability distribution, t - class label
    
    assert_eq!(s.shape.1, 1, "Matrix has to be column vector");
    assert!(t < s.shape.0, "Label too big");

    let mut result = s.clone();
    result[(t, 0)] -= 1.;

    result
}

pub fn mat_sigmoid(mut matrix: Matrix) -> Matrix {
    for i in 0..matrix.shape.0 {
        for j in 0..matrix.shape.1 {
            matrix[(i, j)] = sigmoid(matrix[(i, j)]);
        }
    }

    matrix
}

pub fn mat_softmax(mut matrix: Matrix) -> Matrix {
    let mut e_sum = 0.;
    let max = matrix.max();

    for i in 0..matrix.shape.0 {
        for j in 0..matrix.shape.1 {
            let e_val = (matrix[(i, j)] - max).exp();
            matrix[(i, j)] = e_val;
            e_sum += e_val;
        }
    }

    for i in 0..matrix.shape.0 {
        for j in 0..matrix.shape.1 {
            matrix[(i, j)] /= e_sum;
        }
    }

    matrix
}

pub fn gradient_descent(matrix: &Matrix, gradient: &Matrix, lr: f32) -> Matrix {
    assert_eq!(matrix.shape, gradient.shape, "Matrices have to be same shape!");

    let mut result = matrix.clone();

    for i in 0..result.shape.0 {
        for j in 0..result.shape.1 {
            result[(i, j)] -= lr * gradient[(i, j)]
        }
    }

    result
}

pub fn shuffle(n: usize) -> Vec<usize> {
    // Fisher–Yates shuffle 
    // returns a permutation of 0..n range

    let mut rng = rand::rng();
    let mut shuffled = Vec::new();

    for i in 0..n {
        shuffled.push(i);
    }

    for i in 0..n - 1 {
        let j = rng.random_range(i..n);
        shuffled.swap(i, j);
    }

    shuffled
} 