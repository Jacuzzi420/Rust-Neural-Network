pub fn sigmoid(x: f32) -> f32 {
    // sigmoid function to "squish" values into the [0, 1] range
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(a: &Vec<f32>) -> Vec<f32> {
    // returns vector of sigmoid derivative from activation layer
    // a: n x 1 (activations)

    let n = a.len();
    let mut result = Vec::<f32>::with_capacity(n);

    for i in 0..n {
        result.push(a[i] * (1_f32 - a[i]));
    }

    return result;
}

pub fn softmax(v: &Vec<f32>) -> Vec<f32> {
    // sofmax function to get probablity distribution from vector v

    if v.is_empty() {
        return vec![];
    }

    let n = v.len();
    let mut result = Vec::<f32>::with_capacity(n);

    let mut e_sum = 0_f32;
    let max_val = v.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    for i in 0..n {
        let e_val = (v[i] - max_val).exp();
        e_sum += e_val;
        result.push(e_val);
    }

    for i in 0..n {
        result[i] /= e_sum;
    }

    result
}

pub fn xentropy(s: &Vec<f32>, t: usize) -> f32 {
    // cross entropy - cost function for classification
    // s - vector with probability distribution, t - class label

    assert!(t < s.len(), "Label too big");

    return -(s[t].clamp(10e-9_f32, 1_f32)).ln();
}

pub fn xentropy_grad(s: &Vec<f32>, t: usize) -> Vec<f32> {
    // cross entropy gradient
    // s - vector with probability distribution, t - class label
    
    let mut result = s.clone();
    result[t] -= 1_f32;
    return result;
}
 
pub fn matrix_by_vector(m: &Vec<Vec<f32>>, v: &Vec<f32>) -> Vec<f32> {
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

pub fn transpose(m: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    // transpose matrix M
    // M: n x m

    let n = m.len();
    let p = m[0].len();

    let mut result = Vec::<Vec<f32>>::with_capacity(p);

    for i in 0..p {
        result.push(Vec::<f32>::with_capacity(n));
        
        for j in 0..n {
            result[i].push(m[j][i]);
        }
    }

    return result;
}

pub fn gradient_descent(m1: &mut Vec<Vec<f32>>, m2: Vec<Vec<f32>>, lr: f32) {
    // gradient descent on matrix
    // m1: n x m, m2: n x m
    // lr - learning rate

    let n = m1.len();
    let m = m1[0].len();

    assert_eq!(n, m2.len(), "Matrices have to be same shape!");
    assert_eq!(m, m2[0].len(), "Matrices have to be same shape!");

    for i in 0..n {
        for j in 0..m {
            m1[i][j] -= lr * m2[i][j];
        }
    }
}

pub fn gradient_descent_vector(v1: &mut Vec<f32>, v2: Vec<f32>, lr: f32) {
    // gradient descent on vector
    // v1: n x 1, v2: n x 1
    // lr - learning rate

    let n = v1.len();

    assert_eq!(n, v2.len(), "Vectors have to be same length!");

    for i in 0..n {
        v1[i] -= lr * v2[i];
    }
}

pub fn vec_by_vec_transposed(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<Vec<f32>> {
    // multiply vector v1 by transpose of vector v2
    // v1: n x 1; v2: m x 1
    // result n x m matrix

    let n = v1.len();
    let m = v2.len();

    let mut result = Vec::<Vec<f32>>::with_capacity(n);

    for i in 0..n {
        result.push(Vec::<f32>::with_capacity(m));

        for j in 0..m {
            result[i].push(v1[i] * v2[j]);
        }
    }

    return result;
}

pub fn add_vectors(u: &Vec<f32>, v: &Vec<f32>) -> Vec<f32> {
    // add vectors U and V

    let n = u.len();

    assert_eq!(n, v.len(), "Vectors must have the same length!");

    let mut result = Vec::<f32>::new();

    for i in 0..n {
        result.push(u[i] + v[i]);
    }
    
    result
}

pub fn vec_mul(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
    // multiplies vector v1 by vector v2 elementwise
    // v1: n x 1; v2: n x 1

    let n = v1.len();

    assert_eq!(n, v2.len(), "Vectors must have the same length!");

    let mut result = Vec::<f32>::with_capacity(n);

    for i in 0..n {
        result.push(v1[i] * v2[i]);
    }

    return result;
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