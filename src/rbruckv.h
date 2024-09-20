/*
 * r_radix_bruck.h
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#ifndef SRC_RBRUCKV_H_
#define SRC_RBRUCKV_H_

#include "../rbrucks.h"

extern double init_time, findMax_time, rotateIndex_time, alcCopy_time,
getBlock_time, prepData_time, excgMeta_time, excgData_time, replace_time,
orgData_time, prepSP_time, SP_time;

extern double intra_time;
extern double* iteration_time;

int myPow(int x, unsigned int p);
std::vector<int> convert10tob(int w, int N, int b);
int check_errors(int *recvcounts, long long *recv_buffer, int rank, int nprocs);

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts, int *sdispls,
							  MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
							  MPI_Datatype recvtype, MPI_Comm comm);

int uniform_spreadout_twolayer(int n, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype,
							   char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_inverse_isplit_r_bruck(int n, int r1, int r2, char *sendbuf, int sendcount,
									MPI_Datatype sendtype, char *recvbuf, int recvcount,
									MPI_Datatype recvtype,  MPI_Comm comm);

int TTPL_rbruck_alltoallv(int n, int r, char *sendbuf, int *sendcounts, int *sdispls,
									   MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
									   MPI_Datatype recvtype, MPI_Comm comm);

int TTPL_BT_alltoallv(int n, int r, int bblock, char *sendbuf, int *sendcounts, int *sdispls,
									   MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
									   MPI_Datatype recvtype, MPI_Comm comm);

int TTPL_BT_alltoallv_s1(int n, int r, int bblock, char *sendbuf, int *sendcounts, int *sdispls,
									   MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
									   MPI_Datatype recvtype, MPI_Comm comm);

int TTPL_BT_alltoallv_s2(int n, int r, int bblock, char *sendbuf, int *sendcounts, int *sdispls,
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

int exclisive_or_alltoallv(char *sendbuf, int *sendcounts,
					       int *sdispls, MPI_Datatype sendtype, char *recvbuf,
						   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int twolayer_communicator_linear(int n, char *sendbuf, int *sendcounts, int *sdispls,
								 MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
								 MPI_Datatype recvtype, MPI_Comm comm);

int twolayer_communicator_linear_s2(int n, int bblock1, int bblock2, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int twolayer_communicator_linear_s3(int n, int bblock1, int bblock2, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int twolayer_communicator_linear_s4(int block, int n, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int ML_benchmark(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
		   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);


int twophase_rbruck_alltoallv_om(int r, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

#endif /* SRC_RBRUCKV_H_ */
