#include <armadillo>
#include "impex.h"
#include <cstddef>

bool 
write_matrix(double * mat, int rows, int cols) {
  //convert matrix to armadillo matrix
  arma::mat wrapper(mat, rows, cols);
  arma::diskio::save_arma_binary(wrapper, "testfile");
}


