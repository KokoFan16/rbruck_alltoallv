/*
 * r_radix_bruck.h
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#ifndef SRC_RBRUCKV_H_
#define SRC_RBRUCKV_H_

#include "../rbrucks.h"

int myPow(int x, unsigned int p);
std::vector<int> convert10tob(int w, int N, int b);

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

#endif /* SRC_RBRUCKV_H_ */
