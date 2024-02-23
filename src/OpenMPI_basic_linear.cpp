/*
 * OpenMPI_basic_linear.cpp
 *
 *  Created on: Feb 20, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

int ompi_alltoallv_intra_basic_linear(char *sendbuf, int *sendcounts, int *sdispls,
									  MPI_Datatype sendtype, char *recvbuf, int *recvcounts,
									  int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)

{
    int i, size, rank, err, nreqs;
    char *psnd, *prcv;
    int sext, rext;
    MPI_Request *preq;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Get extent of send and recv types */
    MPI_Type_size(sendtype, &sext);
    MPI_Type_size(recvtype, &rext);

    /* Simple optimization - handle send to self first */
    psnd = (char *) sendbuf + sdispls[rank] * sext;
    prcv = (char *) recvbuf + rdispls[rank] * rext;
    memcpy(prcv, psnd, recvcounts[rank]*rext);

    if (sendcounts[rank] < 0) { return -1; }

    /* If only one process, we're done. */
    if (1 == size) { return MPI_SUCCESS; }

    /* Now, initiate all send/recv to/from others. */
    nreqs = 0;
    preq = (MPI_Request *)malloc(2 * (size-1) * sizeof(MPI_Request));

    /* Post all receives first */
    for (i = 0; i < size; ++i) {
        if (i == rank) { continue; }

        if (0 < recvcounts[i]) {
            prcv = (char *) (recvbuf + rdispls[i] * rext);
            err = MPI_Irecv(prcv, recvcounts[i], recvtype, i, 0, comm, &preq[nreqs]);
            if (MPI_SUCCESS != err) { return -1; }
           	nreqs++;
        }
    }

    /* Now post all sends */
    for (i = 0; i < size; ++i) {
        if (i == rank) { continue; }

        if (0 < sendcounts[i]) {
            psnd = (char *) (sendbuf + sdispls[i] * sext);
            err = MPI_Isend(psnd, sendcounts[i], sendtype, i, 0, comm, &preq[nreqs]);
            if (MPI_SUCCESS != err) { return -1; }
            nreqs++;
        }
    }

    /* Wait for them all.  If there's an error, note that we don't care
     * what the error was -- just that there *was* an error.  The PML
     * will finish all requests, even if one or more of them fail.
     * i.e., by the end of this call, all the requests are free-able.
     * So free them anyway -- even if there was an error, and return the
     * error after we free everything. */
    err = MPI_Waitall(nreqs, preq, MPI_STATUSES_IGNORE);
    /* Free the requests in all cases as they are persistent */
    free(preq);

    return MPI_SUCCESS;
}

