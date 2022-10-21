/*
 * Wrapper for the Eigen library.
 */

#ifndef REHEATFUNQ_PDTB_EIGENWRAP
#define REHEATFUNQ_PDTB_EIGENWRAP

#include <eigen3/Eigen/Dense>

namespace pdtoolbox {

typedef Eigen::Matrix<double,4,4> SquareMatrix;
typedef Eigen::Matrix<double,4,1> ColumnVector;

}

#endif