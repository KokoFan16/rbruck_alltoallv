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

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts, int *sdispls,
							  MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
							  MPI_Datatype recvtype, MPI_Comm comm);

int uniform_spreadout_twolayer(int n, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype,
							   char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_inverse_isplit_r_bruck(int n, int r1, int r2, char *sendbuf, int sendcount,
									MPI_Datatype sendtype, char *recvbuf, int recvcount,
									MPI_Datatype recvtype,  MPI_Comm comm);

int twophase_twolayer_rbruck_alltoallv(int n, int r, char *sendbuf, int *sendcounts, int *sdispls,
									   MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
									   MPI_Datatype recvtype, MPI_Comm comm);

int twolayer_communicator_linear(int n, char *sendbuf, int *sendcounts, int *sdispls,
								 MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
								 MPI_Datatype recvtype, MPI_Comm comm);

int MPICH_intra_scattered(int bblock, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
						  char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int ompi_alltoallv_intra_basic_linear(char *sendbuf, int *sendcounts, int *sdispls,
									  MPI_Datatype sendtype, char *recvbuf, int *recvcounts,
									  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int ompi_alltoallv_intra_pairwise(char *sendbuf, int *sendcounts, int *sdispls,
									  MPI_Datatype sendtype, char *recvbuf, int *recvcounts,
									  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

#endif /* SRC_RBRUCKV_H_ */
