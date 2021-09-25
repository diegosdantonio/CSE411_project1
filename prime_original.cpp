/**
 * CSE 411
 * Fall 2021
 * Project #1 -- Part #1
 *
 * The purpose of this part is to assess the efficiency of TBB.
 * This file implements a naive O(n) prime checker for integers 1--N.
 * We are not going to try to change the algorithm to improve performance.
 * Instead, we will try to parallelize it using three different approaches:
 *
 * 1- TBB parallel_for.
 * 2- Manually using fork-join (std::thread) with static load balancing.
 * 3- Manually using fork-join (std::thread) with dynamic load balancing.
 *
 * Keep in mind that for small problems, it's likely that different
 * approaches perform similarly.  But for N = 10ˆ6 and 10ˆ9, it is possible
 * to get a HUGE speedup and the differences between approaches should arise.
 *
 */

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>
#include <numeric>
#include <atomic>
#include <thread>
#include <exception>
#include "tbb/tbb.h"
//#include <oneapi/tbb/mutex.h>
#include "oneapi/tbb.h"
#include "oneapi/tbb/mutex.h"
#include <oneapi/tbb/task_arena.h>
//#include "tbb/task_scheduler_init.h"


#define TTBB 0
#define TSTA 1
#define TDYN 2

#include "matplotlib-cpp/matplotlibcpp.h"


namespace plt = matplotlibcpp;
using namespace std::chrono;
using namespace std::this_thread;

using namespace std;
using namespace tbb;

// mutex guards
std::mutex std_mutex;
//tbb::mutex tbb_mutex;

///////// Is it Prime number? ////////////
bool isPrime_void(int n)
{
    // 0 and 1 are not prime numbers
    if (n == 0 || n == 1) {
        return false;
    }
    else {
        for (long int i = 2; i <= n / 2; ++i) {
            if (n % i == 0) {
                return false;
               break;
            }
        }
    }
    return true;
}

/* //////////// Function without optimization ////////////
   returned time
*/

 double without_optimization(int n, bool printt){
    auto t1 = std::chrono::high_resolution_clock::now();
    for (long int j = 2; j <= n; j++) {
        bool isPrime = isPrime_void(j);
            if (isPrime && printt) cout << j << " is a prime number \n";
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    return fp_ms.count();
}

/* //////////// Function tbb_function ////////////
   returned time
*/
double tbb_test(int threads_num, long int n, bool printt){
    auto t1 = std::chrono::high_resolution_clock::now();
   // tbb::task_scheduler_init init(threads_num);
    tbb::parallel_for(tbb::blocked_range<long int>(1, n),
                      [&](tbb::blocked_range<long int> r) {
                          for (long int j = r.begin(); j != r.end(); j++) {
                              bool isPrime = isPrime_void(j);
                              if (isPrime && printt) cout << j << " is a prime number \n";
                          }
                      });
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    return fp_ms.count();
}

/* //////////// Function static balance ////////////
   returned time
*/


double staTest(int threads_num, vector<int> &primeNums, int upperbound) {
    // start time
    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<thread> threadPool;

    // fill in the thread pool with threads and start
    for(long int i=0; i<threads_num; i++){
        // i needs to be captured by value otherwise j will be messed up
        threadPool.push_back(thread([&primeNums, i, upperbound, threads_num](){
            for (long int j = 1+i; j < upperbound; j+=threads_num) {
                if(isPrime_void(j)) {
                    std::lock_guard<std::mutex> guard(std::mutex);
                    primeNums.push_back(j);
                }
            }
        }));
    }

    // wait for threads to merge
    for (auto& th : threadPool) th.join();

    // measure the elapsed time using chrono lib
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    // std::cout << "number of primes found: " << primeNums.j << '\n';

    return fp_ms.count();

}


/* //////////// Function dynamic balance ////////////
   returned time
*/

double dynTest(int threads_num, vector<int> &primeNums, int upperbound) {
    // start time
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<thread> threadPool;
    // atomic int
    std::atomic<int> num(1);

    for(long int i=0; i<threads_num; i++){
        threadPool.push_back(thread([&](){
            while (true) {
                long int number = num++;
                if(number>=upperbound) break;
                if(isPrime_void(number)) {
                    std::lock_guard<std::mutex> guard(std::mutex);
                    primeNums.push_back(number);
                }
            }
        }));
    }
    for (auto& th : threadPool) th.join();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    return fp_ms.count();
    // std::cout << "number of primes found: " << primeNums.size() << '\n';
}


/* //////////// Main function ////////////
   returned time
*/
int main() {
    int i, j, k;
    static tick_count t_start, t_end;
    bool isPrime;
    double fp_ms;


//    cout << "Insert N:";
//    cin >> n;
    long int n=10e3;

    // int threads_num = 2;

    vector<double> y(n);
    vector<double> x(n);

    vector<int> threads_num{2, 4,8,12};
    const char *color[3] = {"b-", "r-", "g-"};
    // vector<double> nn{10, 100,10e2,10e3,10e4,10e5,10e6,10e7,10e8,10e9}; //,1000000000
    vector<double> nn{10, 100,1000,1e6};
    vector<int> primeNums(n);
    vector<double> x1(nn.size());
    vector<double> x2(nn.size());
    vector<double> x3(nn.size());
    vector<double> x4(nn.size());    

    bool printt = false;

    for (int j = 0; j < threads_num.size(); ++j)
    {
        for (int i = 0; i < nn.size(); ++i)
        {

        n = nn[i]; 
        cout << "prime number: " << n << "   Number of thread: " << threads_num[j] << "\n";

        // fp_ms = without_optimization(n, printt);
        // std::cout << "elapsed time: " << fp_ms << " ms" << "\n";
        fp_ms = tbb_test(threads_num[j], n, printt);
        std::cout << "TBB elapsed time: " << fp_ms << " ms" << '\n';        
        x1[i] = fp_ms;

        }

        std::string var = std::to_string(threads_num[j]);
        plt::loglog(nn,x2,color[j]); 
        plt::named_plot("Number of thread " + var,nn,x1,color[j]);
        plt::xlabel("n");
        plt::ylabel("Time(ms)");
        plt::grid(true);
    //    plt::save("fig1.png");

        // std::string var = std::to_string(threads_num[j]);

        // plt::title("TBB prime number test");
        plt::legend();
        
    }
    plt::save("fig1.png");
    plt::clf();

    for (int j = 0; j < threads_num.size(); ++j)
    {
        for (int i = 0; i <= nn.size(); ++i)
        {

        n = nn[i]; 
        cout << "prime number: " << n << "   Number of thread: " << threads_num[j] << "\n";

        // fp_ms = without_optimization(n, printt);
        // std::cout << "elapsed time: " << fp_ms << " ms" << "\n";
        fp_ms = staTest(threads_num[j], primeNums, x.size());
        std::cout << "Static balance elapsed time: " << fp_ms << " ms" << '\n';       
        x1[i] = fp_ms;


        }

        std::string var = std::to_string(threads_num[j]);
        plt::loglog(nn,x2,color[j]); 
        plt::named_plot("Number of thread " + var,nn,x1,color[j]);
        plt::xlabel("n");
        plt::ylabel("Time(ms)");
        plt::grid(true);
    //    plt::save("fig1.png");

        // std::string var = std::to_string(threads_num[j]);

        // plt::title("Static balance prime number test");
        plt::legend();
        
    }
    plt::save("fig2.png");
    plt::clf();



    for (int j = 0; j < threads_num.size(); ++j)
    {
        for (int i = 0; i <= nn.size(); ++i)
        {

        n = nn[i]; 
        cout << "prime number: " << n << "   Number of thread: " << threads_num[j] << "\n";

        // fp_ms = without_optimization(n, printt);
        // std::cout << "elapsed time: " << fp_ms << " ms" << "\n";
        // fp_ms = staTest(threads_num[j], primeNums, x.size());
        // std::cout << "Static balance elapsed time: " << fp_ms << " ms" << '\n';       
        fp_ms = dynTest(threads_num[j], primeNums, x.size());
        std::cout << "Dynamic balance elapsed time: " << fp_ms << " ms" << "\n \n";
        x1[i] = fp_ms;


        }

        std::string var = std::to_string(threads_num[j]);
        plt::loglog(nn,x2,color[j]); 
        plt::named_plot("Number of thread " + var,nn,x1,color[j]);
        plt::xlabel("n");
        plt::ylabel("Time(ms)");
        plt::grid(true);
    //    plt::save("fig1.png");

        // std::string var = std::to_string(threads_num[j]);

        // plt::title("Static balance prime number test");
        plt::legend();
        
    }
    plt::save("fig3.png");
    plt::clf();

    // fp_ms = tbb_test(threads_num, n, printt);
    // std::cout << "TBB elapsed time: " << fp_ms << " ms" << '\n';
    // x2[i] = fp_ms;

    // fp_ms = staTest(threads_num, primeNums, x.size());
    // std::cout << "Static balance elapsed time: " << fp_ms << " ms" << '\n';
    // x3[i] = fp_ms;

    // fp_ms = dynTest(threads_num, primeNums, x.size());
    // std::cout << "Dynamic balance elapsed time: " << fp_ms << " ms" << "\n \n";
    // x4[i] = fp_ms;

    // threads_num = threads_num*2;
    



//    x[j] = j;
//    y[j] = fp_ms.count();
//    cout << y[y.size()-1];
    
    // plt::named_plot("Without optimization",nn, x1);

//     plt::semilogy(nn,x2,"b-"); 
//     plt::semilogy(nn,x3,"r-");
//     plt::semilogy(nn,x4,"g-");

//     plt::named_plot("TTB",nn,x2,"b-");
//     plt::named_plot("Static thread",nn,x3,"r-");
//     plt::named_plot("Dynamic thread",nn,x4,"g-");
    
   
//     plt::xlabel("n");
//     plt::ylabel("Time(ms)");
//     plt::grid(true);
// //    plt::save("fig1.png");

//     std::string var = std::to_string(threads_num);

//     plt::title("Number of thread " + var);
//     plt::legend();
    // plt::show();
    return 0;

}