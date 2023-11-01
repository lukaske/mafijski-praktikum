#include <iostream>
#include <cmath>
#include <cpp/Eigen>

using namespace std;
using namespace Eigen;

double off_diag_norm(const MatrixXd& A) {
    int n = A.rows();
    double dia_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        dia_sum += A(i, i) * A(i, i);
    }
    return A.array().square().sum() - dia_sum;
}

pair<MatrixXd, MatrixXd> Jacobi(MatrixXd A, double precision = 1e-10, int max_iter = 1000) {
    int n = A.rows();
    MatrixXd A_ = A;
    MatrixXd D_ = MatrixXd::Identity(n, n);
    double S = off_diag_norm(A_);
    int i = 0;

    while (S > precision && i < max_iter) {
        cout << "Iteracija " << i << ", S = " << S << endl;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double theta = (A_(q, q) - A_(p, p)) / (2 * A_(p, q));
                double t = copysign(1.0, theta) / (abs(theta) + sqrt(theta * theta + 1));
                double c = 1.0 / sqrt(t * t + 1);
                double s = t * c;
                MatrixXd J = MatrixXd::Identity(n, n);
                J(p, p) = c;
                J(q, q) = c;
                J(p, q) = s;
                J(q, p) = -s;
                S -= 2 * A_(p, q) * A_(p, q);
                D_ = D_ * J;
                A_ = J.transpose() * A_ * J;
                i += 1;
            }
        }
        cout << "Iteracija " << i << ", S = " << S << endl;
    }

    return make_pair(A_, D_);
}

int main() {
    MatrixXd A(4, 4);
    A << 4, 1, 2, 0,
         1, 3, 0, 1,
         2, 0, 2, 2,
         0, 1, 2, 3;

    double precision = 1e-10;
    int max_iter = 1000;
    pair<MatrixXd, MatrixXd> result = Jacobi(A, precision, max_iter);
    
    cout << "Resulting A matrix:" << endl << result.first << endl;
    cout << "Resulting D matrix:" << endl << result.second << endl;
    
    return 0;
}
