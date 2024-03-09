/*
 * utils.cpp
 *
 *      Author: kokofan
 */

#include "rbruckv.h"

int myPow(int x, unsigned int p) {
  if (p == 0) return 1;
  if (p == 1) return x;

  int tmp = myPow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}


std::vector<int> convert10tob(int w, int N, int b) {

	std::vector<int> v(w);
	int i = 0;
	while(N) {
	  v[i++] = (N % b);
	  N /= b;
	}
//	std::reverse(v.begin(), v.end());
	return v;
}

int check_errors(int *recvcounts, long long *recv_buffer, int rank, int nprocs){
	// check correctness
	int error = 0, index = 0;
	for (int p = 0; p < nprocs; p++) {
		for (int s = 0; s < recvcounts[p]; s++) {
			if ( p != 0 || rank != 0) {
				if (recv_buffer[index] == 0) {
					error++;
				}
			}
			if ( (recv_buffer[index] % 10) != (rank % 10) ) { error++; }
			index++;
		}
	}
	return error;
}



