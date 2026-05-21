use rand::prelude::*;
use std::ops::{Index, IndexMut};
use std::{fmt::{Display, Formatter}};

// TODO: Zamień na funkcje na operatory
#[derive(Clone)]
pub struct Matrix {
    pub shape: (usize, usize),
    data: Vec<f32>,
}

impl Matrix {
    pub fn new(shape: (usize, usize), data: Vec<f32>) -> Matrix {
        assert_eq!(data.len(), shape.0 * shape.1);
        Matrix { shape, data }
    }

    pub fn zeros(shape: (usize, usize)) -> Matrix {
        assert_ne!(shape.0 * shape.1, 0);

        let data = vec![0.0; shape.0 * shape.1];
        Matrix { shape, data }
    }

    pub fn rand(shape: (usize, usize), low: f32, high: f32) -> Matrix {
        assert_ne!(shape.0 * shape.1, 0);

        let n = shape.0 * shape.1;
        let mut rng = rand::rng();
        let mut data = Vec::<f32>::with_capacity(n);
        for _ in 0..n {
            data.push(rng.random_range(low..high));
        }

        Matrix { shape, data }
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros((self.shape.1, self.shape.0));

        for i in 0..result.shape.0 {
            for j in 0..result.shape.1 {
                result[(i, j)] = self[(j, i)];
            }
        }

        result
    }

    pub fn mat_mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape.1, other.shape.0);

        let n = self.shape.1;
        let mut result = Matrix::zeros((self.shape.0, other.shape.1));

        for i in 0..result.shape.0 {
            for j in 0..result.shape.1 {
                for k in 0..n {
                    result[(i, j)] += self[(i, k)] * other[(k, j)];
                }
            }
        }

        result
    }

    pub fn scalar_mul(&self, scalar: f32) -> Matrix {
        let mut result = Matrix::zeros(self.shape);

        for i in 0..result.shape.0 {
            for j in 0..result.shape.1 {
                result[(i, j)] = self[(i, j)] * scalar;
            }
        }

        result
    }

    pub fn max(&self) -> f32 {
        let mut max = f32::NEG_INFINITY;

        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                if self[(i, j)] > max {
                    max = self[(i, j)];
                }
            }
        }

        max
    }

    pub fn min(&self) -> f32 {
        let mut min = f32::INFINITY;

        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                if self[(i, j)] < min {
                    min = self[(i, j)];
                }
            }
        }

        min
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape, other.shape);

        let mut result = Matrix::zeros(self.shape);

        for i in 0..result.shape.0 {
            for j in 0..result.shape.1 {
                result[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }

        result
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape, other.shape);

        let mut result = Matrix::zeros(self.shape);

        for i in 0..result.shape.0 {
            for j in 0..result.shape.1 {
                result[(i, j)] = self[(i, j)] * other[(i, j)];
            }
        }

        result
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (x, y) = index;
        &self.data[x * self.shape.1 + y]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (x, y) = index;
        &mut self.data[x * self.shape.1 + y]
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.shape.0 {
            write!(f, "[")?;
            for j in 0..self.shape.1 {
                if j == self.shape.1 - 1 {
                    write!(f, "{}", self[(i, j)])?;
                }
                else {
                    write!(f, "{} ", self[(i, j)])?;
                }
            }
            if i == self.shape.0 - 1 {
                write!(f, "]]")?;
            }
            else {
                write!(f, "]\n")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_index() {
        let mut matrix = Matrix::new((2, 5), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        assert_eq!(matrix.data[1 * matrix.shape.1 + 2], matrix[(1, 2)]);

        matrix[(1, 2)] = 67.;
        assert_eq!(matrix.data[1 * matrix.shape.1 + 2], 67.);
    }

    #[test]
    fn test_matrix_from_vector() {
        let matrix = Matrix::new((2, 5), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        println!("{}", matrix);
    }

    #[test]
    fn test_matrix_zeros() {
        let matrix = Matrix::zeros((3, 4));
        println!("{}", matrix);
    }

    #[test]
    fn test_matrix_rand() {
        let matrix = Matrix::rand((5, 4), 0.0, 1.0);
        println!("{}", matrix);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix::new((2, 5), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        println!("{}", matrix);
        println!("{}", matrix.transpose())
    }
}