#[cfg(test)]
use super::*;

#[test]
fn test_matrix_by_vector() {
    // test matrix_by_vector function

    let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let v = vec![3.0, 2.0, 1.0];
    assert_eq!(mlp::matrix_by_vector(&m, &v), vec![10.0, 28.0]);
}

#[test]
fn test_add_vectors() {
    // test add_vectors function

    let u = vec![-0.5, 2.6, 4.0];
    let v = vec![0.0, 0.4, -5.0];
    assert_eq!(mlp::add_vectors(&u, &v), vec![-0.5, 3.0, -1.0]);
}
