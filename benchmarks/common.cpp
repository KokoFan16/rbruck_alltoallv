/*
 * MLinputs_example.cpp
 *
 *  Created on: Feb 5, 2024
 *      Author: kokofan
 */

#include "../src/rbruckv.h"

void spreadout_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
		   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));
	for (int i = 0; i < nprocs; i++) {
		int src = (rank + i) % nprocs; // avoid always to reach first master node

		MPI_Irecv((char *)recvbuf + rdispls[src] * typesize, recvcounts[src], recvtype, src, 0, comm, &req[i]);
	}

	for (int i = 0; i < nprocs; i++) {
		int dst = (rank - i + nprocs) % nprocs;

		MPI_Isend((char *) sendbuf + sdispls[dst] * typesize, sendcounts[dst], sendtype, dst, 0, comm, &req[i+nprocs]);
	}

	MPI_Waitall(2*nprocs, req, stat);
	free(req);
	free(stat);
}



