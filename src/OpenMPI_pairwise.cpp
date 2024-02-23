/*
 * OpenMPI_pairwise.cpp
 *
 *  Created on: Feb 20, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

int ompi_alltoallv_intra_pairwise(char *sendbuf, int *sendcounts, int *sdispls,
									  MPI_Datatype sendtype, char *recvbuf, int *recvcounts,
									  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
    int line = -1, err = 0, rank, size, step = 0, sendto, recvfrom;
    int sdtype_size, rdtype_size;
    void *psnd, *prcv;
    MPI_Request req;
    MPI_Aint sext, rext;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Get extent of send and recv types */
    MPI_Type_size(sendtype, &sdtype_size);
    MPI_Type_size(recvtype, &rdtype_size);

    MPI_Type_extent(sendtype, &sext);
    MPI_Type_extent(sendtype, &rext);


   /* Perform pairwise exchange starting from 1 since local exchange is done */
    for (step = 0; step < size; step++) {
        req = MPI_REQUEST_NULL;

        /* Determine sender and receiver for this step. */
        sendto  = (rank + step) % size;
        recvfrom = (rank + size - step) % size;

        /* Determine sending and receiving locations */
        psnd = (char*)sendbuf + sdispls[sendto] * sext;
        prcv = (char*)recvbuf + rdispls[recvfrom] * rext;

        /* send and receive */
        if (0 < recvcounts[recvfrom] && 0 < rdtype_size) {
            err = MPI_Irecv(prcv, recvcounts[recvfrom], recvtype, recvfrom, 0, comm, &req);
            if (MPI_SUCCESS != err) { return -1; }
        }

        if (0 < sendcounts[sendto] && 0 < sdtype_size) {
            err = MPI_Send(psnd, sendcounts[sendto], sendtype, sendto, 0, comm);
            if (MPI_SUCCESS != err) { return -1; }
        }

        if (MPI_REQUEST_NULL != req) {
            err = MPI_Wait(&req, MPI_STATUS_IGNORE);
            if (MPI_SUCCESS != err) { return -1; }
        }
    }

	return MPI_SUCCESS;
}



