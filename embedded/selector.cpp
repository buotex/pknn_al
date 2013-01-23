#include <armadillo>
#include <random>
#include <algorithm>
int __getSampleIndex(const arma::vec & data, double alpha) {
//rescale data
  
  int lastIndex = alpha * data.n_elem - 1;
  double max = data[0];
  double treshold = data[lastIndex];
  arma::vec probabilities = (data.rows(0,lastIndex) -  treshold) / (max - treshold);
  probabilities /= arma::accu(probabilities);
  //probabilities.print("probabilities");
  
  arma::vec summed_probabilities = arma::cumsum(probabilities);

  //summed_probabilities.print("summed");

  srand((unsigned)time(NULL));
  double X=((double)rand()/(double)RAND_MAX);
  int index =  std::lower_bound(summed_probabilities.begin(), summed_probabilities.end(), X) -  summed_probabilities.begin();
  return index;
   
}


extern "C" {
  int getSampleIndex(double * data, int n_elem, double alpha) {
    arma::vec vec(data, n_elem, false);
    arma::uvec indices = arma::sort_index(vec, 1);
    int index = __getSampleIndex(arma::sort(vec, 1), alpha);
    return indices[index];
  }

}
