/**
 * CSE 411
 * Fall 2020
 * Project #1 -- Part #2
 *
 * The purpose of this assignment is to make correct code run faster.  This file
 * (gauss.cc) file implements the O(n^3) Gaussian Elimination algorithm.  We are
 * not going to try to change the algorithm to improve performance.  Instead, we
 * will try to use a variety of techniques, to include improving locality, using
 * SIMD operations, and exploiting multicore, to accelerate the program.
 *
 * Keep in mind that for small problems, it's likely not possible to get a
 * parallel speedup.  But for 2048 and 4096, it is possible to get a HUGE
 * speedup.
 *
 * Instructions
 * - We will be using Intel's Threading Building Blocks (tbb) to take advantage
 *   of multiple cores.  You will need to install libtbb-dev in order to use TBB.
 *
 * - If you are using Docker:
 *   chances are good that Docker on your laptop won't let you access all the
 *   cores of your laptop.  At some point, you'll need to test your code on the
 *   sunlab.  When you do that, you'll need to update your Makefile accordingly.
 *   You will probably also need to manually place a copy of TBB in your home
 *   folder on sunlab.
 *
 *  - To use GCC 7 on sunlab, type
 *      module load gcc-7.1.0
 *
 * - It is wise to vary the grainsize in TBB.  You may add a command-line
 *   parameter for that purpose.
 *
 * - TBB allows specifying the number of threads.  In order to generate charts
 *   that show speedup at different thread counts, you will want to use the
 *   task_scheduler_init object.
 *
 * - You will need to understand C++ lambdas in order to complete
 *   this assignment.  You should also think about numerical stability.
 *   "Pivoting" is essential!
 *
 * - When testing on Sunlab, keep in mind that the machines are shared.  If you
 *   wait until the last minute, you may not have exclusive access to the
 *   machine, and your results will be invalid.
 *
 * - You must produce a 2-3 page write-up of your experience.  Describe the
 *   techniques you used to parallelize the code.  Include graphs showing the
 *   performace for 2048 and 4096 matrices, with threads on the X axis and time
 *   on the Y axis.  Results should be the average of 5 trials, and should
 *   discuss variance.
 *
 * - Turn-in will be via Course Site. Work individually or in groups of max 2
 */

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <random>
#include <atomic>
#include <unistd.h>
#include "tbb/tbb.h"
#include <oneapi/tbb/task_arena.h>
#include "vector"

#include "matplotlib-cpp/matplotlibcpp.h"
//
//
namespace plt = matplotlibcpp;
using namespace tbb;
//using namespace std;

/**
 * matrix_t represents a 2-d (square) array of doubles
 */
class matrix_t {
  /**
   * M is the matrix.  It is an array of arrays, so that we can swap row
   * pointers in O(1) instead of swapping rows in O(n)
   */
  double **M;

  /** the # rows / # columns / sqrt(# elements) */
  unsigned int size;

public:
  /** Construct by allocating the matrix */
  matrix_t(unsigned int n) : M(new double *[n]), size(n) {
    for (unsigned i = 0; i < size; ++i)
      M[i] = new double[size];
  }

  /** Destruct by de-allocating */
  ~matrix_t() {
    for (unsigned i = 0; i < size; ++i)
      delete M[i];
    delete M;
  }

  /** Give the illusion of this being a simple array */
  double *&operator[](std::size_t idx) { return M[idx]; };
  double *const &operator[](std::size_t idx) const { return M[idx]; };
  unsigned int getSize() { return size; }

  /** I doubt the above illusion **/
  double* getPointer(int row, int col) { return &(M[row][col]); }
};

/**
 * vector_t represents a 1-d array of doubles
 */
class vector_t {
  /** simple array of doubles */
  double *V;

  /** size of V */
  unsigned int size;

public:
  /** Construct by allocating the vector */
  vector_t(unsigned int n) : V(new double[n]), size(n) {}

  /** Destruct by freeing the vector */
  ~vector_t() { delete V; }

  /** Give the illusion of this being a simple array */
  double &operator[](std::size_t idx) { return V[idx]; };
  const double &operator[](std::size_t idx) const { return V[idx]; };
  unsigned int getSize() { return size; }

  /** I doubt the above illusion **/
  double* getPointer(int idx) { return &(V[idx]); }
};

/**
 * Given a random seed, populate the elements of A and then B with a
 * sequence of random numbers in the range (-range...range)
 */
void initializeFromSeed(int seed, matrix_t &A, vector_t &B,
                        unsigned int range) {
  // Use a Mersenne Twister to create doubles in the requested range
  double drange = 1.0*range;
  std::mt19937 seeder(seed);
  auto mt_rand =
      std::bind(std::uniform_real_distribution<double>(-drange, drange),
                seeder);

  // populate A
  for (unsigned i = 0; i < A.getSize(); ++i)
    for (unsigned j = 0; j < A.getSize(); ++j)
      A[i][j] = (double)(mt_rand());

  // populate B
  for (unsigned i = 0; i < B.getSize(); ++i)
    B[i] = (double)(mt_rand());
}

/** Print the matrix and array in a form that looks good */
void print(matrix_t &A, vector_t &B) {
  for (unsigned i = 0; i < A.getSize(); ++i) {
    for (unsigned j = 0; j < A.getSize(); ++j)
      std::cout << std::setw(15) << A[i][j];
    std::cout << " | " << std::setw(15) << B[i] << "\n";
  }
  std::cout << std::endl;
}

/**
 * parallel version of the gauss elimination
 */
void gauss_parallel(matrix_t &A, vector_t &B, vector_t &X, int grain_size) {
  // iterate over rows
  for (unsigned i = 0; i < A.getSize(); ++i) {
    // NB: we are now on the ith column

    // For numerical stability, find the largest value in this column
    // in parallel
    std::atomic<double> big(abs(A[i][i]));
    std::atomic<unsigned> row(i);
    parallel_for(blocked_range<unsigned>(i + 1, A.getSize(), grain_size), \
      [&](blocked_range<unsigned> &r){
        for (unsigned k = r.begin(); k != r.end(); ++k) {
          if (abs(A[k][i]) > big) {
            big = abs(A[k][i]);
            row = k;
          }
        }
      }
    );

    // Given our random initialization, singular matrices are possible!
    if (big == 0.0) {
      std::cout << "The matrix is singular!" << std::endl;
      exit(-1);
    }
    // swap so max column value is in ith row
    std::swap(A[i], A[row]);
    std::swap(B[i], B[row]);
    // Eliminate the ith row from all subsequent rows: this sounds like locality
    //
    // NB: this will lead to all subsequent rows having a 0 in the ith
    // column
    parallel_for(blocked_range<unsigned>(i+1, A.getSize(), grain_size),\
      [&](const blocked_range<unsigned> &r){
        for (unsigned k = r.begin(); k != r.end(); ++k) {
          double c = -A[k][i] / A[i][i];
          parallel_for(blocked_range<unsigned>(i, A.getSize(), grain_size),\
            [&](const blocked_range<unsigned> &s){
              for (unsigned j = s.begin(); j != s.end(); ++j){
                if (i == j){
                  A[k][j] = 0;
                } else{
                  A[k][j] += c * A[i][j];
                }
              }
            }
          );
          B[k] += c * B[i];
        }
      }
    );
  }

  for (unsigned i = 0; i < A.getSize(); ++i){
    /***************************************************************************
    ***** The following code block was supposed to work but it did not
    // parallel_for(blocked_range2d<unsigned>(i+1, A.getSize(), grain_size,\
    //   i, A.getSize(), grain_size),\
    //   [&](const blocked_range2d<unsigned> &r){
    //     tbb_mutex.lock();
    //     tbb_mutex.unlock();
    //     for (unsigned k = r.rows().begin(); k != r.rows().end(); ++k) {
    //       double c = -A[k][i] / A[i][i];
    //       tbb_mutex.lock();
    //       std::cout << k << '\n';
    //       tbb_mutex.unlock();
    //       for (unsigned j = r.cols().begin(); j != A.getSize(); ++j){
    //         if (i == j){
    //           A[k][j] = 0;
    //         } else{
    //           A[k][j] += c * A[i][j];
    //         }
    //       }
    //       B[k] += c * B[i];
    //     }
    //   }
    // );
    ***************************************************************************/
  }

  // NB: A is now an upper triangular matrix

  // Use back substitution to solve equation A * x = b
  for (int i = A.getSize() - 1; i >= 0; --i) {
    X[i] = B[i] / A[i][i];
    // found locality here:
    parallel_for(blocked_range<int>(0, i, grain_size),
      [&](blocked_range<int> &r){
        for (int k = r.begin(); k!=r.end(); ++k){
          B[k] -= A[k][i] * X[i];
        }
      }
    );
  }
}

/**
 * For a system of equations A * x = b, with Matrix A and Vectors B and X,
 * and assuming we only know A and b, compute x via the Gaussian Elimination
 * technique
 */
void gauss(matrix_t &A, vector_t &B, vector_t &X) {
  // iterate over columns
  for (unsigned i = 0; i < A.getSize(); ++i) {
    // NB: we are now on the ith column

    // For numerical stability, find the largest value in this column
    double big = abs(A[i][i]);
    int row = i;
    for (unsigned k = i + 1; k < A.getSize(); ++k) {
      if (abs(A[k][i]) > big) {
        big = abs(A[k][i]);
        row = k;
      }
    }
    // Given our random initialization, singular matrices are possible!
    if (big == 0.0) {
      std::cout << "The matrix is singular!" << std::endl;
      exit(-1);
    }

    // swap so max column value is in ith row
    std::swap(A[i], A[row]);
    std::swap(B[i], B[row]);

    // Eliminate the ith row from all subsequent rows
    //
    // NB: this will lead to all subsequent rows having a 0 in the ith
    // column
    for (unsigned k = i + 1; k < A.getSize(); ++k) {
      double c = -A[k][i] / A[i][i];
      for (unsigned j = i; j < A.getSize(); ++j)
        if (i == j)
          A[k][j] = 0;
        else
          A[k][j] += c * A[i][j];
      B[k] += c * B[i];
    }
  }

  // NB: A is now an upper triangular matrix

  // Use back substitution to solve equation A * x = b
  for (int i = A.getSize() - 1; i >= 0; --i) {
    X[i] = B[i] / A[i][i];
    for (int k = i - 1; k >= 0; --k)
      B[k] -= A[k][i] * X[i];
  }
}

/**
 * Make sure that the values in X actually satisfy the equation A * x = b
 *
 * Unfortunately, this check isn't so simple.  Even with double precision
 * floating point, we lose some significant digits, and thus a naive check
 * won't pass.
 */
void check(matrix_t &A, vector_t &B, vector_t &X) {
  vector_t V(B.getSize());
  parallel_for(blocked_range<unsigned>(0, A.getSize()),
    [&](blocked_range<unsigned> &q){
      for (unsigned i = q.begin(); i != q.end(); ++i) {
        // compute the value of B based on X
        V[i] = parallel_reduce(
          blocked_range<unsigned>(0, A.getSize()), double(0), \
            [&](blocked_range<unsigned> &r, double init){
              for (unsigned j = r.begin(); j != r.end(); ++j)
                init += A[i][j] * X[j];
              return init;
            },
            std::plus<double>()
        );
      }
    }
  );

  // we can't just compare ans to B[i].  But if the two are close, then
  // their ratio will compute to 1 even at double precision
  for(unsigned i=0; i<V.getSize(); ++i){
    double ratio = std::max(abs(V[i] / B[i]), abs(B[i] / V[i]));
    if (ratio != 1) {
      std::cout << "Verification failed for index = " << i << "." << std::endl;
      std::cout << V[i] << " != " << B[i] << std::endl;
      return;
    }
  }
  std::cout << "Verification succeeded" << std::endl;
}

/** Print some helpful usage information */
void usage() {
  using std::cout;
  cout << "Gaussian Elimination Solver\n";
  cout << "  Usage: gauss [options]\n";
  cout << "    -r <int> : seed for the random number generator (default 411)\n";
  cout << "    -n <int> : number of rows in the matrix (default 256)\n";
  cout << "    -g <int> : range for values in the matrix (default 65536)\n";
  cout << "    -t <int> : number of threads (default 4)\n";
  cout << "    -v       : toggle verbose output (default false)\n";
  cout << "    -p       : toggle parallel mode (default false)\n";
  cout << "    -c       : toggle verifying the result (default true)\n";
  cout << "    -h       : print this message\n";
}

int main(int argc, char *argv[]) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;

  // Config vars that we get via getopt


  int seed = 411; // random seed

  const char *color[5] = {"b-", "r-", "g-","k-","m-"};

  int size = 400; // # rows in the matrix
  std::vector<int> size2 = { 3, 20, 30, 150, 300, 600, 1000, 2048, 4096};
//    std::vector<int> size2 = { 3, 20, 30, 150};
    std::vector<double> y1(size2.size());
    std::vector<double> x1(size2.size());

  int range = 65536; // matrix elements will have values between -range and range
  int threads = 4;
    std::vector<int> threads2 = {2,4,8,16,32,64,128,256};
//    std::vector<double> y1(threads2.size());
//    std::vector<double> x1(threads2.size());
    int grain_size = 1;
    std::vector<int> grain_size2 = {1,2,4,8,16};

  bool verbose = false;  // should we print some diagnostics?
  bool docheck = true;   // should we verify the output?
  bool parallel = false; // use parallelism?



    for (int j = 0; j < grain_size2.size(); ++j) {
//        threads = threads2[j];

        grain_size = grain_size2[j];
        for (int i = 0; i < threads2.size(); ++i) {
//            size = size2[i];
//            x1[i] =
            threads = threads2[i];
            x1[i] = threads;
            // Print the configuration... this makes results of scripted experiments
            // much easier to parse
            std::cout << "r,n,g,p,t,a = " << seed << ", " << size << ", " << range << ", "
                      << parallel << ", " << threads << ", " << grain_size << std::endl;

            // the number of numThreads
            //  tbb::task_scheduler_init init(threads);
            oneapi::tbb::task_arena arena(threads);

            // Create our matrix and vectors, and populate them with default values
            matrix_t A(size);
            vector_t B(size);
            vector_t X(size);
            initializeFromSeed(seed, A, B, range);

            // Print initial matrix
            if (verbose) {
                std::cout << "Matrix (A) | B" << std::endl;
                print(A, B);
            }

            // Calculate solution
            auto starttime = high_resolution_clock::now();
            if (parallel)
                gauss_parallel(A, B, X, grain_size);
            else
                gauss(A, B, X);
            auto endtime = high_resolution_clock::now();


            // Check the solution?
            if (docheck) {
                // Pseudorandom number generators are nice... We can re-create A and
                // B by re-initializing them from the same seed as before
                initializeFromSeed(seed, A, B, range);
                check(A, B, X);
            }

            // Print the execution time
            duration<double> time_span = duration_cast<duration<double>>(endtime - starttime);
            std::cout << "Total execution time:\t\t\t\t\t" << time_span.count() << " seconds" << std::endl;


            y1[i] = time_span.count();
        }
        std::string var = std::to_string(grain_size2[j]);
        plt::loglog(x1,y1,color[j]);
        plt::named_plot("Grain size " + var,x1,y1,color[j]);
    }
//    plt::loglog(size2,y1);
//    plt::xlabel("Number of thread");
    plt::ylabel("Time(ms)");
    plt::legend();
    plt::save("fig6.png");
//    plt::clf();
    plt::show();
}
