#include <vigra/random_forest.hxx>
#include <armadillo>
#include <iostream>
#include <cstdio>
#include "stdint.h"

using namespace vigra;
using namespace rf;

//label, count

void 
countVotes(const MultiArrayView<2,double> & kernel, const MultiArrayView<2, double> &
    fullLabels, int numClasses, const double * trnIndData, int trnN, const double * queIndData,
    int queN, int numForests, uint32_t * output)   {
  std::vector<RandomForest<> > forests;

  typedef MultiArray<2,double> Array;
  typedef MultiArray<2,uint32_t> IArray;
  for (int i = 0; i < numForests; ++i) {

    //use 2 times the objects, number of training data as the number of
    //dimensions.
    MultiArray<2, double> data(Array::size_type(2*trnN, trnN));
    IArray labels(Array::size_type(2*trnN, 1));



    for (int k = 0; k < trnN; ++k) {
      int randomInt = rand() % queN;
      for (int l = 0; l < trnN; ++l) {

        data(k, l) = kernel(trnIndData[k], trnIndData[l]);
        data(k+trnN, l) = kernel(queIndData[randomInt], trnIndData[l]);
      } 
      labels(k) = fullLabels(trnIndData[k]);
      labels(k+trnN) = 0;

    }
    arma::mat test(data.data(),data.shape(0), data.shape(1), false);
//    test.print("test");
    if (!test.is_finite()) {
      printf("NaN in data\n");
    }
    //std::cout << "finite" <<  test.is_finite() << std::endl;

    RandomForest<> rf(RandomForestOptions().tree_count(1));

    visitors::OOB_Error oob_v;

    arma::umat testlabels(labels.data(), labels.shape(0), labels.shape(1), false);
    //testlabels.print("testlabels");

    rf.learn(data, labels, visitors::create_visitor(oob_v));
    //std::cout <<"the out-of-bag error is: " << oob_v.oob_breiman << "\n";

    forests.push_back(rf);


  }

  //counting
  Array queries(Array::size_type(queN, trnN));
  for (int i = 0; i < queN; ++i) {
    for (int j = 0; j < trnN; ++j) {
      queries(i, j) = kernel(queIndData[i], trnIndData[j]);
    }
  }
 // arma::mat test(queries.data(), queries.shape(0), queries.shape(1));
  //test.print("queries");

  for (std::size_t j = 0; j < forests.size(); ++j) {
    IArray pLabels(Array::size_type(queN, 1));
    //Array rawLabels(Array::size_type(queN, numClasses + 1));
    forests[j].predictLabels(queries, pLabels);
    //forests[j].predictRaw(queries, rawLabels);

    for (std::size_t k = 0; k < pLabels.size(); ++k) { 
      output[j * queN + k ] = pLabels(k);
      //counts.push_back(pLabels(k));
    }
    //arma::imat test(pLabels.data(), pLabels.shape(0), pLabels.shape(1));
    //test.print("plabels");
    //arma::mat test(rawLabels.data(), rawLabels.shape(0), rawLabels.shape(1));
    //test.print("rawLabels");

  }

}

template <typename T>
MultiArrayView<2, T>
wrapArray( T* array, int m, int n) {

  typedef MultiArrayView<2,T> Array;
  MultiArrayView<2, T> data(typename Array::size_type(m,n), array);
  return data;
}
template <typename T>
MultiArrayView<1, T>
wrapArray( T * array, int m) {
  TinyVector<int,1> shape(m);
  MultiArrayView<1, T> data(shape, array);
  return data;
}


