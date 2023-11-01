#include <iostream>
#include <cmath>
#include <Eigen>
#include <chrono>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sstream>

#include "json.hpp"

using json = nlohmann::json;
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

std::pair<int, int> find_pivot(const Eigen::MatrixXd& A) {
    int n = A.rows();
    Eigen::MatrixXd A_ = A;

    for (int i = 0; i < n; i++) {
        A_(i, i) = 0;
    }

    int max_idx = 0;
    double max_value = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double abs_value = std::abs(A_(i, j));
            if (abs_value > max_value) {
                max_value = abs_value;
                max_idx = i * n + j;
            }
        }
    }

    int p = max_idx / n;
    int q = max_idx % n;

    return std::make_pair(p, q);
}

pair<MatrixXd, MatrixXd> Jacobi(MatrixXd A, double precision = 1e-2, int max_iter = 5) {
    int n = A.rows();
    MatrixXd A_ = A;
    MatrixXd D_ = MatrixXd::Identity(n, n);
    double S = off_diag_norm(A_);
    int i = 0;

    while (i < max_iter and S > precision) {
        for (int p_i = 0; p_i < n - 1; ++p_i) {
            for (int q_i = p_i + 1; q_i < n; ++q_i) {
                pair<int,int> pivot = find_pivot(A_);
                int p = pivot.first;
                int q = pivot.second;
                double t;
                if (abs(A_(p, q)) < precision) {
                    t = 0.0;
                }
                double theta = (A_(q, q) - A_(p, p)) / (2 * A_(p, q));
                t = copysign(1.0, theta) / (abs(theta) + sqrt(theta * theta + 1));
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
            }
        }
        double S = off_diag_norm(A_);
        i += 1;
    }

    cout << "N: " << n << " iters: " << i << " best S: " << S << endl;

    return make_pair(A_, D_);
}

static std::string toString(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

int main() {

    // Read the JSON data from a file
    std::ifstream f("../test_data.json");
    json jsonData = json::parse(f);
    vector<MatrixXd> matrices;

    for (int i = 0; i < 98; i++) {
        int n = jsonData["N"][to_string(i)];
        vector<vector<float>> A = jsonData.at("flattened_matrix").at(to_string(i));
        Eigen::MatrixXd eigenMatrix(n, n);
        for (int i = 0; i < eigenMatrix.rows(); i++) {
            for (int j = 0; j < eigenMatrix.cols(); j++) {
                eigenMatrix(i, j) = A[i][j];
            }
        }
        matrices.push_back(eigenMatrix);
    }

    vector<MatrixXd> eigenvalues;
    vector<float> iter_time;
    double total_time = 0;
    
    
    typedef std::chrono::nanoseconds ns;

    for (int i = 0; i < 98; i = i + 5) {
        //cout << "Matrix " << i + 2 << endl;
        auto start = chrono::high_resolution_clock::now();
        pair<MatrixXd, MatrixXd> result = Jacobi(matrices[i]);
        auto end = std::chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        ns d = chrono::duration_cast<ns>(duration);
        double time = d.count() / 1e6;
        total_time += time;
        iter_time.push_back(time);
        eigenvalues.push_back(result.first);

        cout << "Execution time: " << i << " is " << time << " milliseconds" << endl;
    }

    cout << "Execution time: " << total_time << " milliseconds" << endl;

    // To JSON file

    // Create a JSON object to store the results
    json resultJSON;

    // Store eigenvalues as a JSON array of arrays (vector<vector<float>>)
    json eigenvaluesJSON;
    for (const Eigen::MatrixXd& eigenMatrix : eigenvalues) {
        vector<vector<double>> eigenMatrixData;
        for (int i = 0; i < eigenMatrix.rows(); i++) {
            vector<double> row;
            for (int j = 0; j < eigenMatrix.cols(); j++) {
                row.push_back(static_cast<double>(eigenMatrix(i, j)));
            }
            eigenMatrixData.push_back(row);
        }
        eigenvaluesJSON.push_back(eigenMatrixData);
    }

    // Add matrices, iter_time, and total_time to the JSON object
    resultJSON["total_time"] = total_time;
    resultJSON["iter_time"] = iter_time;
    resultJSON["matrices"] = eigenvaluesJSON;

    // Write the JSON to a file
    std::ofstream outputFile("cpp_benchmark.json");
    outputFile << resultJSON.dump(4);
    outputFile.close();


    
    return 0;
}
