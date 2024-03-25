/*
 * exclusive_or.cpp
 *
 *  Created on: Feb 20, 2024
 *      Author: kokofan
 */


#include "rbruckv.h"

 /* This algorithm only works when P is power of 2 */
int exclisive_or_alltoallv(char *sendbuf, int *sendcounts,
					       int *sdispls, MPI_Datatype sendtype, char *recvbuf,
						   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
    int rank, size, src, dst, step;
    int sdtype_size, rdtype_size;
    void *psnd, *prcv;
    MPI_Aint sext, rext;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Check if P is power of 2 */
    int w = ceil(log(size) / float(log(2)));

//    if (rank == 0) {
//		std::cout << "Math -- exclisive-or: " << size << " " << log(size) << " " <<  log(2) << " " << w << " " << myPow(2, w) << std::endl;
//	}

    if (size != myPow(2, w)) { return -1; }

    /* Get extent of send and recv types */
    MPI_Type_size(sendtype, &sdtype_size);
    MPI_Type_size(recvtype, &rdtype_size);

    MPI_Type_extent(sendtype, &sext);
    MPI_Type_extent(sendtype, &rext);

    for (step = 0; step < size; step++) {
    	src = dst = rank ^ step;
    	psnd = (char *) (sendbuf + sdispls[dst]*sext);
    	prcv = (char *) (recvbuf + rdispls[src]*rext);
    	MPI_Sendrecv(psnd, sendcounts[dst]*sext, MPI_CHAR, dst, 0,
    			prcv, recvcounts[src]*rext, MPI_CHAR, src, 0, comm, MPI_STATUS_IGNORE);
    }

	return MPI_SUCCESS;
}


