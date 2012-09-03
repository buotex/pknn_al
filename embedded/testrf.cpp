#include <armadillo>
#include <string>
#include "randomforest.h"

arma::mat
loadMatrix(const char * filename)  {
  arma::mat matrix;
  std::string err_msg;
  arma::diskio::load_arma_binary(matrix, std::string(filename), err_msg);

  return matrix;

}

int main() {


  arma::mat kernel = loadMatrix("kernel");
  arma::mat labels = loadMatrix("labels");




  //arma::umat labels = arma::conv_to<arma::umat>::from(dlabels);

  std::vector<double> trainingIndices;
  std::vector<double> queryIndices;
  for (int i = 0; i < 60; ++i) {
    trainingIndices.push_back(i);
  }
  for (int i = trainingIndices.size(); i < kernel.n_rows; ++i) {
    queryIndices.push_back(i);
  }


  MultiArrayView<2, double> wkernel = wrapArray(kernel.memptr(), kernel.n_rows, kernel.n_cols);
  MultiArrayView<2, double> wlabels = wrapArray(labels.memptr(), labels.n_rows, 1);

  int numForests = 1;
  int numClasses = 40;
  std::vector<uint32_t> output(numForests * queryIndices.size());
  countVotes(
      wkernel,
      wlabels, 
      numClasses,
      &(trainingIndices[0]),
      trainingIndices.size(),
      &(queryIndices[0]),
      queryIndices.size(),
      numForests,
      &(output[0])
      );
  std::cout << numForests << " " << trainingIndices.size();
  arma::umat testcounts(&(output[0]), numForests, queryIndices.size(), false);
  testcounts.t().print("test");
}
