/*
 * twolayer_communicator_linear.cpp
 *
 *  Created on: Feb 19, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

int twolayer_communicator_linear(int n, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	int rank, nprocs, sendsize, recvsize, color;
    MPI_Comm intra_comm;
    char *recvaddr, *sendaddr;
    int *intra_sendcounts, *intra_sdispls, *intra_recvcounts, *intra_rdispls;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

	if (nprocs % n < 0 || n >= nprocs) {
		if	(rank == 0)
			std::cout << "ERROR: the process count should be divided by the process count of a group." << std::endl;
		 MPI_Abort(comm, -1);
	}


    MPI_Type_size(sendtype, &sendsize);
    MPI_Type_size(recvtype, &recvsize);

    // Assuming an even number of processes, split into two groups
    color = rank/n; // Group identifier
    MPI_Comm_split(comm, color, rank, &intra_comm);

    intra_sendcounts = (int *) sendcounts + n*color;
    intra_recvcounts = (int *) recvcounts + n*color;
    intra_sdispls = (int *) sdispls + n*color;
    intra_rdispls = (int *) rdispls + n*color;

    // intra-node alltoallv
    MPICH_intra_scattered(0, sendbuf, intra_sendcounts, intra_sdispls, sendtype, recvbuf, intra_recvcounts, intra_rdispls, recvtype, intra_comm);

	int group_rank, group_size, ngroup, nquest=0;
    MPI_Comm_rank(intra_comm, &group_rank);
    MPI_Comm_size(intra_comm, &group_size);

    ngroup = ceil(nprocs / float(n)); // number of groups

	MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));

    for (int i = 1; i < ngroup; i++) {

    	int ndst = (color + i) % ngroup;
    	int nsrc = (color - i + ngroup) % ngroup;

    	if (nsrc != color) {
    		for (int j = 0; j < n; j++) {
    			int src =  nsrc * n + j;
    			if (src < nprocs) {
					recvaddr = (char *) (recvbuf + rdispls[src] * recvsize);
					MPI_Irecv(recvaddr, recvcounts[src]*recvsize, MPI_CHAR, src, 0, comm, &req[nquest++]);
    			}
    		}
    	}

    	if (ndst != color) {
    		for (int j = 0; j < n; j++) {
    			int dst = ndst * n + j;
    			if (dst < nprocs) {
					sendaddr = (char *) sendbuf + sdispls[dst] * sendsize;
					MPI_Isend(sendaddr, sendcounts[dst]*sendsize, MPI_CHAR, dst, 0, comm, &req[nquest++]);
    			}
    		}
    	}
    }

	MPI_Waitall(nquest, req, stat);
	free(req);
	free(stat);

    MPI_Comm_free(&intra_comm);

	return 0;
}



int twolayer_communicator_linear_s2(int n, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	int rank, nprocs, sendsize, recvsize, color;
    MPI_Comm intra_comm;
    char *recvaddr, *sendaddr;
    int *intra_sendcounts, *intra_sdispls, *intra_recvcounts, *intra_rdispls;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

	if (nprocs % n < 0 || n >= nprocs) {
		if	(rank == 0)
			std::cout << "ERROR: the process count should be divided by the process count of a group." << std::endl;
		 MPI_Abort(comm, -1);
	}


    MPI_Type_size(sendtype, &sendsize);
    MPI_Type_size(recvtype, &recvsize);

    // Assuming an even number of processes, split into two groups
    color = rank/n; // Group identifier
    MPI_Comm_split(comm, color, rank, &intra_comm);

    intra_sendcounts = (int *) sendcounts + n*color;
    intra_recvcounts = (int *) recvcounts + n*color;
    intra_sdispls = (int *) sdispls + n*color;
    intra_rdispls = (int *) rdispls + n*color;

    // intra-node alltoallv
    MPICH_intra_scattered(0, sendbuf, intra_sendcounts, intra_sdispls, sendtype, recvbuf, intra_recvcounts, intra_rdispls, recvtype, intra_comm);

	int group_rank, ngroup, nquest=0;
    MPI_Comm_rank(intra_comm, &group_rank);

    ngroup = ceil(nprocs / float(n)); // number of groups

	MPI_Request* req = (MPI_Request*)malloc(2*n*(ngroup-1)*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*n*(ngroup-1)*sizeof(MPI_Status));

    for (int i = 1; i < ngroup; i++) {

    	int ndst = (color + i) % ngroup;
    	int nsrc = (color - i + ngroup) % ngroup;

    	if (nsrc != color) {
    		for (int j = 0; j < n; j++) {
    			int src =  nsrc * n + ((j + group_rank) % n);
    			if (src < nprocs) {
    				recvaddr = (char *) recvbuf + rdispls[src] * recvsize;
    				MPI_Irecv(recvaddr, recvcounts[src]*recvsize, MPI_CHAR, src, 0, comm, &req[nquest++]);
    			}
    		}
    	}

    	if (ndst != color) {
    		for (int j = 0; j < n; j++) {
    			int dst = ndst * n + ((j - group_rank + n) % n);
    			if (dst < nprocs) {
					sendaddr = (char *) sendbuf + sdispls[dst] * sendsize;
					MPI_Isend(sendaddr, sendcounts[dst]*sendsize, MPI_CHAR, dst, 0, comm, &req[nquest++]);
    			}
    		}
    	}
    }

	MPI_Waitall(nquest, req, stat);

	free(req);
	free(stat);

    MPI_Comm_free(&intra_comm);

	return 0;
}



int twolayer_communicator_linear_s3(int n, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	int rank, nprocs, sendsize, recvsize;
	int ngroup, gid, grank, intrap;
    char *recvaddr, *sendaddr;
    int *intra_sendcounts, *intra_sdispls, *intra_recvcounts, *intra_rdispls;
    int src, dst, rp, sp, nquest=0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

	if (nprocs % n < 0 || n >= nprocs) {
		if	(rank == 0)
			std::cout << "ERROR: the process count should be divided by the process count of a group." << std::endl;
		 MPI_Abort(comm, -1);
	}

    MPI_Type_size(sendtype, &sendsize);
    MPI_Type_size(recvtype, &recvsize);

    ngroup = ceil(nprocs / float(n)); // number of groups
    gid = rank / n; // Group identifier
    grank = rank % n; // local rank in each group

	intrap = n*gid;

	/* Intra-node Comm */
	MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));
	for (int i = 0; i < n; i++) {
		src  = (grank + i) % n;
		rp = intrap + src;
		if (rp < nprocs) {
			recvaddr = (char *) recvbuf + rdispls[rp] * recvsize;
			MPI_Irecv(recvaddr, recvcounts[rp]*recvsize, MPI_CHAR, rp, 0, comm, &req[nquest++]);
		}
	}
	for (int i = 0; i < n; i++) {
		dst = (grank - i + n) % n;
		sp = intrap + dst;
		if (sp < nprocs) {
			sendaddr = (char *) sendbuf + sdispls[sp] * sendsize;
			MPI_Isend(sendaddr, sendcounts[sp]*sendsize, MPI_CHAR, sp, 0, comm, &req[nquest++]);
		}
	}
	MPI_Waitall(nquest, req, stat);

	/* Inter-node Comm */
	nquest = 0;
    for (int i = 1; i < ngroup; i++) {

    	int ndst = (gid + i) % ngroup;
    	int nsrc = (gid - i + ngroup) % ngroup;

    	if (nsrc != gid) {
    		for (int j = 0; j < n; j++) {
    			int src =  nsrc * n + ((j + grank) % n);
    			if (src < nprocs) {
    				recvaddr = (char *) recvbuf + rdispls[src] * recvsize;
    				MPI_Irecv(recvaddr, recvcounts[src]*recvsize, MPI_CHAR, src, 0, comm, &req[nquest++]);
    			}
    		}
    	}

    	if (ndst != gid) {
    		for (int j = 0; j < n; j++) {
    			int dst = ndst * n + ((j - grank + n) % n);
    			if (dst < nprocs) {
					sendaddr = (char *) sendbuf + sdispls[dst] * sendsize;
					MPI_Isend(sendaddr, sendcounts[dst]*sendsize, MPI_CHAR, dst, 0, comm, &req[nquest++]);
    			}
    		}
    	}
    }

	MPI_Waitall(nquest, req, stat);


	free(req);
	free(stat);

	return 0;
}


