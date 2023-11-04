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

static std::string toString(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

std::vector<float> cumulativeSum(const std::vector<float>& input) {
    std::vector<float> result;
    float sum = 0.0f;

    for (const float& value : input) {
        sum += value;
        result.push_back(sum);
    }

    return result;
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
    vector<int> N;
    vector<double> num_precision;
    vector<float> iter_time;
    double total_time = 0;
    
    
    typedef std::chrono::nanoseconds ns;

    for (int i = 0; i < 98; i = i + 5) {
        auto start = chrono::high_resolution_clock::now();
        SelfAdjointEigenSolver<MatrixXd> eigensolver(matrices[i]);
        if (eigensolver.info() != Success) {
            cerr << "Eigenvalue decomposition failed." << endl;
            return 1;
        }
        MatrixXd result = eigensolver.eigenvectors();
        auto end = std::chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        ns d = chrono::duration_cast<ns>(duration);
        double time = d.count() / 1e6;
        total_time += time;
        iter_time.push_back(time);
        eigenvalues.push_back(result);
        num_precision.push_back(off_diag_norm(result));
        N.push_back(i+2);
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
    resultJSON["total_time"] = cumulativeSum(iter_time);
    resultJSON["iter_time"] = iter_time;
    resultJSON["matrices"] = eigenvaluesJSON;
    resultJSON["num_precision"] = num_precision;
    resultJSON["N"] = N;
    // Write the JSON to a file
    std::ofstream outputFile("eigen_benchmark.json");
    outputFile << resultJSON.dump(4);
    outputFile.close();
    
    return 0;
}
