/*
 * MLinputs_example.cpp
 *
 *  Created on: Feb 5, 2024
 *      Author: kokofan
 */

#include "../src/rbruckv.h"
#include <fstream>
#include <sstream>

int nprocs, rank;
int run(int loopcount, int csize, int lsize, int warmup);
void split_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
		   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

struct {
    double value;
    int rank;
} local, global;

int main(int argc, char **argv) {

    if (argc < 4) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <csize> <lsize>" << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);
    int csize = atoi(argv[2]);
    int lsize = atoi(argv[3]);


    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");


    run(loopCount, csize, lsize, 0);

    MPI_Finalize();
    return 0;
}

int run(int loopcount, int csize, int lsize, int warmup) {

	int sendcounts[nprocs], recvcounts[nprocs], sdispls[nprocs], rdispls[nprocs];
	int soffset = 0, roffset = 0, error = 0;

	// Uniform random distribution
	for (int i=0; i < nprocs; i++) { sendcounts[i] = csize; }
	if (rank == 0) { sendcounts[1] = lsize; }

	// Initial send offset array
	for (int i = 0; i < nprocs; ++i) {
		sdispls[i] = soffset;
		soffset += sendcounts[i];
	}

	// Initial receive counts and offset array
	MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

	for (int i = 0; i < nprocs; ++i) {
		rdispls[i] = roffset;
		roffset += recvcounts[i];
	}

	// Initial send buffer
	long long* send_buffer = new long long[soffset];
	long long* recv_buffer = new long long[roffset];

	int index = 0;
	for (int i = 0; i < nprocs; i++) {
		for (int j = 0; j < sendcounts[i]; j++)
			send_buffer[index++] = i + rank * 10;
	}

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < loopcount; t++) {

    	double start = MPI_Wtime();
    	spreadout_alltoallv((char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
    			(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    	double end = MPI_Wtime();
    	double comm_time = (end - start);

    	local.value = comm_time;
    	local.rank = rank;

    	if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

			if (rank == global.rank) {
				std::cout << "Spread-out " << nprocs << " " << rank << " "
						<< comm_time << " " << csize << " " << lsize << std::endl;
			}
    	}
    }

	MPI_Barrier(MPI_COMM_WORLD);

	if (csize != lsize) {
		for (int t = 0; t < loopcount; t++) {
			double start = MPI_Wtime();
			split_alltoallv((char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double end = MPI_Wtime();
			double comm_time = (end - start);

			// check correctness
			error = check_errors(recvcounts, recv_buffer, rank, nprocs);
			if (error > 0) {
				std::cout << "[Split] has errors for " << csize << " " << lsize << std::endl;
			}

			local.value = comm_time;
			local.rank = rank;

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

				if (rank == global.rank) {
					std::cout << "Split " << nprocs << " " << rank << " "
							<< comm_time << " " << csize << " " << lsize << std::endl;
				}
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	delete[] send_buffer;
	delete[] recv_buffer;

	return 0;
}


void split_alltoallv(char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
		   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	int typesize, div_size = 0, nreqs = 0;
	MPI_Type_size(sendtype, &typesize);

	if (rank == 0) { div_size = sendcounts[1] / nprocs; }
	if (rank == 1) { div_size = recvcounts[0] / nprocs; }

	MPI_Request* req = (MPI_Request*)malloc(3*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(3*nprocs*sizeof(MPI_Status));

	for (int i = 0; i < nprocs; i++) {

		if (rank == 1) {
			// Process 1 receives chunks from process 0 in each round
			MPI_Irecv((char *)recvbuf + (rdispls[0] + i * div_size) * typesize,
					  div_size, recvtype, 0, 0, comm, &req[nreqs++]);
		}

		if (rank == 1 && i != 0) {
			// Process 1 receives from all ranks except process 0
			MPI_Irecv((char *)recvbuf + rdispls[i] * typesize, recvcounts[i],
					  recvtype, i, 0, comm, &req[nreqs++]);

		} else if (rank != 1) {
			// All other processes communicate normally with all ranks
			int src = (rank - i + nprocs) % nprocs;
			MPI_Irecv((char *)recvbuf + rdispls[src] * typesize, recvcounts[src],
					  recvtype, src, 0, comm, &req[nreqs++]);
		}
	}


	for (int i = 0; i < nprocs; i++) {

		// Process 0 sends chunks to process 1 in each round
		if (rank == 0) {
			MPI_Isend((char *)sendbuf + (sdispls[1] + i * div_size) * typesize,
					  div_size, sendtype, 1, 0, comm, &req[nreqs++]);
		}

		if (rank == 0 && i != 1) {
			// Process 0 sends to and receives from all ranks except process 1
			MPI_Isend((char *)sendbuf + sdispls[i] * typesize, sendcounts[i],
					  sendtype, i, 0, comm, &req[nreqs++]);

		} else if (rank != 0) {
			// All other processes communicate normally with all ranks
			int dst = (rank + i) % nprocs;
			MPI_Isend((char *)sendbuf + sdispls[dst] * typesize, sendcounts[dst],
					  sendtype, dst, 0, comm, &req[nreqs++]);
		}
	}

	MPI_Waitall(nreqs, req, stat);
	free(req);
	free(stat);
}







