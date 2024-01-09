/*
 * r_radix_bruck.h
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#ifndef SRC_RADIX_R_BRUCK_H_
#define SRC_RADIX_R_BRUCK_H_

#include "../rbrucks.h"

extern double intra_time;
extern double inter_time;

int myPow(int x, unsigned int p);
std::vector<int> convert10tob(int w, int N, int b);
void calculate_commsteps_and_datablock_counts(int nprocs, int r, std::vector<int>& the_sd_pstep);

void uniform_radix_r_bruck(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_norotation_radix_r_bruck(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_norotation_radix_r_bruck_dt(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_modified_radix_r_bruck(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_modified_inverse_r_bruck(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_isplit_r_bruck(int n, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_norot_radix_r_bruck(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void optimized_radix_r_bruck(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void spreadout_alltoall(char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_inverse_isplit_r_bruck(int n, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void intra_communication_test(int n, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void inter_communication_test(int n, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

void uniform_inverse_isplit_r_bruck(int n, int r1, int r2, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm);

#endif /* SRC_RADIX_R_BRUCK_H_ */
