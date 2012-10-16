//what we need here is a random generator, which produces a vector (?) of
//randomly generated numbers, according to the needed distribution.


#include <random>
#include <functional>
#include <algorithm>
#include <armadillo>
#include <cstdio>

//Howto: For a given length of alpha-variables, produce one gamma distribution
//

typedef struct {
double p1;
double p2;
} p_struct;

p_struct createDirich(std::vector<double> alpha, std::size_t numSamples) {
  arma::mat R(numSamples, alpha.size()); //careful because of fortran <-> c-order!
  typedef std::gamma_distribution<double> Distribution;
  std::default_random_engine rSeedEngine_;
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    auto generator = std::bind(Distribution(alpha[i]), std::ref(rSeedEngine_));
    std::generate(R.begin() + numSamples * i, R.begin() + numSamples * (i + 1), generator); 

  }

  //normalize for every sample:
  arma::mat scaling = arma::sum(R,1);
  for (std::size_t i = 0; i < numSamples; ++i) {
    R.row(i) /= arma::as_scalar(scaling(i));
  }
  p_struct p;
  p.p2 = 1 - arma::as_scalar(arma::sum(arma::max(R, 1))) / numSamples;
  arma::mat alphaW(alpha.data(), alpha.size(), 1, false);
  p.p1 = 1 - arma::as_scalar(arma::max(alphaW) / arma::sum(alphaW));
  //arma::sum(R, 1).print();
  return p;
}
arma::mat createDistri(std::vector<double> alpha, std::size_t numSamples) {



}
extern "C" {

  p_struct
      createDirich(double * alpha, size_t length, size_t numSamples) {
        return createDirich(std::vector<double>(alpha, alpha + length), numSamples);
      }

}







int main() {

  std::vector<double> alpha = {1.0,2.0,5.0,0.5};
  createDirich(alpha, 1000);

}
