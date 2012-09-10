#include <random>
#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
 
//For sampling, let's try about 1000? thetas.

std::vector<double> getThetas (std::size_t N, const std::vector<double>& alpha ) {
  std::mt19937 eng(time(NULL));
  std::vector<gamma_distribution<double> > distributions;
  //First create one gamma distribution per alpha
  for (std::size_t i = 0; i < alpha.size(); ++i ) {
    distributions.push_back(std::gamma_distribution<double>(alpha[i]));         
  }
  //Create, for the N needed sample, N values per gamma-distribution
  std::vector<double> thetas(N * alpha.size());
  for (std::size_t i = 0; i < N; ++i ) {
    double sum = 0;
    for (std::size_t j = 0; j < alpha.size(); ++j ) {
      double val = distributions[j](eng);
      thetas[i * alpha.size() + j] = val;
      sum += val;
    }
    for (std::size_t j = 0; j < alpha.size(); ++j ) {
      thetas[i * alpha.size() + j] /= sum;
    }
  }
  return thetas;
}


std::vector<double> evaluateDirichlets(const std::vector<double> & alpha ,const std::vector<double> &thetas) {
  std::size_t k = alpha.size();
  std::size_t numTheta = thetas.size() / k;

  for (std::size_t i = 0 ; i < numTheta; ++i) {
     
  

  }
}

std::vector<double> sum(const std::vector<double> &thetas, std:size_t N, std::size_t L ){
  std::vector<double> sums;
for (std::size_t i = 0; i < N; ++i) {
  double sum = 0;
  double min = std::numeric_limits<double>::max();
  for (std::size_t j = 0; j < L; ++j) {
    double val = thetas[i * L + j];
    sum += val;
    min = std::min(min, val);
  
  }
  sum -= min;
  sums.push_back(sum);
  return sums;
}
