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

    sendaddr = (char *) sendbuf + sdispls[n*color] * sendsize;
    recvaddr = (char *) recvbuf + rdispls[n*color] * recvsize;
    intra_sendcounts = (int *) sendcounts + n*color;
    intra_recvcounts = (int *) recvcounts + n*color;
    intra_sdispls = (int *) sdispls + n*color;
    intra_rdispls = (int *) rdispls + n*color;

//    std::cout << rank << " " << intra_sendcounts[0] << std::endl;

//    std::cout << rank << " " << sdispls[n*color] << " " << rdispls[n*color] << std::endl;

    // intra-node alltoallv
    MPICH_intra_scattered(0, sendaddr, intra_sendcounts, sdispls, sendtype, recvaddr, intra_recvcounts, rdispls, recvtype, intra_comm);

	int group_rank, group_size;
    MPI_Comm_rank(intra_comm, &group_rank);
    MPI_Comm_size(intra_comm, &group_size);

    int ngroup = nprocs / n; // number of groups

//	int nsend[ngroup], nrecv[ngroup], nsdisp[ngroup], nrdisp[ngroup];
//	int soffset = 0, roffset = 0;
//	for (int i = 0; i < ngroup; i++) {
//		nsend[i] = 0, nrecv[i] = 0;
//		for (int j = 0; j < n; j++) {
//			int id = i * n + j;
//			int sn = sendcounts[id];
//			nsend[i] += sn;
//			nrecv[i] += recvcounts[id];
//		}
////		std::cout << rank << " " << nsend[i] << " " << nrecv[i] << std::endl;
//		nsdisp[i] = soffset, nrdisp[i] = roffset;
//		soffset += nsend[i] * sendtype, roffset += nrecv[i] * recvtype;
//	}

//	MPI_Request* req = (MPI_Request*)malloc(2*ngroup*sizeof(MPI_Request));
//	MPI_Status* stat = (MPI_Status*)malloc(2*ngroup*sizeof(MPI_Status));
//	for (int i = 0; i < ngroup; i++) {
//
//		int nsrc = (color + i) % ngroup;
//		int src =  nsrc * n + group_rank; // avoid always to reach first master node
//
//		MPI_Irecv(&recvbuf[nrdisp[nsrc]], nrecv[nsrc]*recvtype, MPI_CHAR, src, 0, comm, &req[i]);
//	}
//
//	for (int i = 0; i < ngroup; i++) {
//		int ndst = (color - i + ngroup) % ngroup;
//		int dst = ndst * n + group_rank;
//
//		MPI_Isend(&sendbuf[nsdisp[ndst]], nsend[ndst]*sendtype, MPI_CHAR, dst, 0, comm, &req[i+ngroup]);
//	}
//
//	MPI_Waitall(2*ngroup, req, stat);
//
//	free(req);
//	free(stat);
//
////
	if (rank == 7) {
		int index = 0;
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < recvcounts[i]; j++){
				long long a;
				memcpy(&a, &recvbuf[index*recvsize], recvsize);
				std::cout << a << std::endl;
				index++;
			}
		}
	}


//	MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
//	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));
//	for (int i = 0; i < nprocs; i++) {
//		int src = (rank + i) % nprocs; // avoid always to reach first master node
//
//		if (rank == 1)
//			std::cout << "send " << rank << " " << i << " " << src << " " << std::endl;
//		MPI_Irecv(&recvbuf[src*recvcount*typesize], recvcount*typesize, MPI_CHAR, src, 0, comm, &req[i]);
//	}
//
//	for (int i = 0; i < nprocs; i++) {
//		int dst = (rank - i + nprocs) % nprocs;
//
//		if (rank == 1)
//			std::cout << "receive " << rank << " " << i << " " << dst << " " << std::endl;
//		MPI_Isend(&sendbuf[dst*sendcount*typesize], sendcount*typesize, MPI_CHAR, dst, 0, comm, &req[i+nprocs]);
//	}
//
//	MPI_Waitall(2*nprocs, req, stat);
//	free(req);
//	free(stat);

    MPI_Comm_free(&intra_comm);

	return 0;
}

