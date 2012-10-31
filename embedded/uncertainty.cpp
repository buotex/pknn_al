//what we need here is a random generator, which produces a vector (?) of
//randomly generated numbers, according to the needed distribution.


#include <random>
#include <functional>
#include <algorithm>
#include <armadillo>
#include <cstdio>
#include <future>

//Howto: For a given length of alpha-variables, produce one gamma distribution
//

typedef std::gamma_distribution<double> Distribution;

typedef struct {
  double p1;
  double p2;
} p_struct;

template <typename InputIterator, typename Generator>
int writeToArray(InputIterator it, InputIterator end, Generator generator) {

  //auto generator = std::bind(Distribution(1), eng);
  //std::cout << it << std::endl;
  std::generate(it, end, generator);
}
p_struct createDirich(std::vector<double> alpha, std::size_t numSamples) {
  arma::mat R(numSamples, alpha.size()); //careful because of fortran <-> c-order!
  typedef std::gamma_distribution<double> Distribution;

  std::vector<int> seeds(alpha.size());
  std::seed_seq({0}).generate(seeds.begin(), seeds.end());

  std::vector<std::future<int> > answers(alpha.size());

  for (std::size_t i = 0; i < alpha.size(); ++i) {
    std::default_random_engine rSeedEngine_(seeds[i]);
    auto generator = std::bind(Distribution(alpha[i]), rSeedEngine_);
    answers[i] = std::async(std::launch::deferred,&writeToArray<decltype(R.begin()), decltype(generator)>, R.begin() + numSamples * i, R.begin() + numSamples * (i + 1), generator);
  }
  for (int i = 0; i < alpha.size(); ++i) {
    answers[i].get();
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
  //std::cout << "blub" << std::endl;
  return p;
}


int createSingleDirich(double * alpha, std::size_t length, std::size_t numSamples, double * results) {
  arma::mat R(numSamples, length); //careful because of fortran <-> c-order!

  for (std::size_t i = 0; i < length; ++i) {
    std::default_random_engine rSeedEngine_((unsigned int) *alpha);
    auto generator = std::bind(Distribution(alpha[i]), rSeedEngine_);
    std::generate(R.begin() + numSamples * i, R.begin() + numSamples * (i + 1), generator);
  }
  //normalize for every sample:
  arma::mat scaling = arma::sum(R,1);
  for (std::size_t i = 0; i < numSamples; ++i) {
    R.row(i) /= arma::as_scalar(scaling(i));
  }
  //p_struct p;

  *(results+1) = 1 - arma::as_scalar(arma::sum(arma::max(R, 1))) / numSamples;
  arma::mat alphaW(alpha, length, 1, false);
  *results = 1 - arma::as_scalar(arma::max(alphaW) / arma::sum(alphaW));
  //arma::sum(R, 1).print();
  //std::cout << "blub" << std::endl;
  return 0;
}
extern "C" {

  p_struct
      createDirich(double * alpha, std::size_t length, std::size_t numSamples) {
        return createDirich(std::vector<double>(alpha, alpha + length), numSamples);
      }
  int 
      createDirichMatrix(double * alpha, std::size_t length, std::size_t numSamples, std::size_t numObjects, double * results) {
        //std::vector<int> seeds(numObjects);
        //std::seed_seq({0}).generate(seeds.begin(), seeds.end());
        std::vector<std::future<int> > answers(numObjects);
        int errorcode = 0;
        for (std::size_t i = 0; i < numObjects; ++i) {
          answers[i] = std::async(std::launch::async,createSingleDirich,alpha + i * length, length, numSamples, results + i * 2);
        }

        for (std::size_t i = 0; i < numObjects; ++i) {
          errorcode += answers[i].get();
        }
        return errorcode;
      }


}


int main() {

  std::vector<double> alpha = {1.0,2.0,5.0,0.5};
  createDirich(alpha, 1000);

}
