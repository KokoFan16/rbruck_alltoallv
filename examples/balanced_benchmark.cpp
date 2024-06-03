/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include <typeinfo>
#include "../src/rbruckv.h"

static int rank, nprocs;
static void run_benckmark(int loopcount, int ncores, int warmup);

int main(int argc, char **argv) {
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    if (argc < 3) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <ncores> " << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);
    int ncores = atoi(argv[2]);
//
//    std::vector<int> bases;
//    for (int i = 2; i < argc; i++)
//    	bases.push_back(atoi(argv[i]));

    // warm-up only
    run_benckmark(5, nprocs, 1);

    // actual running
    run_benckmark(loopCount, ncores, 0);

	MPI_Finalize();
    return 0;
}


static void run_benckmark(int loopcount, int ncores, int warmup) {

	int mpi_errno = MPI_SUCCESS;

	for (int n = 2; n <= 4096; n = n * 2) {

		int sendcounts[nprocs]; // the size of data each process send to other process
		memset(sendcounts, 0, nprocs*sizeof(int));
		int sdispls[nprocs];
		int soffset = 0;

		// Uniform random distribution
		for (int i=0; i < nprocs; i++) {
			sendcounts[i] = 0;
			if (i % ncores == 0) {
				sendcounts[i] = n;
			}
		}

		// Initial send offset array
		for (int i = 0; i < nprocs; ++i) {
			sdispls[i] = soffset;
			soffset += sendcounts[i];
		}

		// Initial receive counts and offset array
		int recvcounts[nprocs];
		MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
		int rdispls[nprocs];
		int roffset = 0;
		for (int i = 0; i < nprocs; ++i) {
			rdispls[i] = roffset;
			roffset += recvcounts[i];
		}

//		if (rank == 0 ) {
//			for (int i = 0; i < nprocs; ++i) {
//				std::cout << sendcounts[i]  << " " << recvcounts[i] << std::endl;
//			}
//		}

		// Initial send buffer
		long long* send_buffer = new long long[soffset];
		long long* recv_buffer = new long long[roffset];

		int index = 0;
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < sendcounts[i]; j++)
				send_buffer[index++] = i + rank * 10;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// MPI_alltoallv
		for (int it = 0; it < loopcount; it++) {
			double st = MPI_Wtime();
			MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double et = MPI_Wtime();
			double total_time = et - st;

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (total_time == max_time)
					std::cout << "[MPIAlltoallv-1] " << nprocs << " " << n << " "<<  max_time << std::endl;
			}
		}

		delete[] send_buffer;
		delete[] recv_buffer;

		MPI_Barrier(MPI_COMM_WORLD);


		memset(sendcounts, 0, nprocs*sizeof(int));
		memset(recvcounts, 0, nprocs*sizeof(int));
		soffset = 0; roffset = 0;
		for (int i=0; i < nprocs; i++) {
			recvcounts[i] = 0;
			if (i % ncores == 0) {
				recvcounts[i] = n;
			}
		}

		// Initial send offset array
		for (int i = 0; i < nprocs; ++i) {
			rdispls[i] = roffset;
			roffset += recvcounts[i];
		}

		// Initial receive counts and offset array
		MPI_Alltoall(recvcounts, 1, MPI_INT, sendcounts, 1, MPI_INT, MPI_COMM_WORLD);
		for (int i = 0; i < nprocs; ++i) {
			sdispls[i] = soffset;
			soffset += sendcounts[i];
		}

		send_buffer = new long long[soffset];
		recv_buffer = new long long[roffset];

		index = 0;
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < sendcounts[i]; j++)
				send_buffer[index++] = i + rank * 10;
		}

//		if (rank == 4 ) {
//			for (int i = 0; i < nprocs; ++i) {
//				std::cout << sendcounts[i]  << " " << recvcounts[i] << std::endl;
//			}
//		}

		for (int it = 0; it < loopcount; it++) {
			double st = MPI_Wtime();
			MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double et = MPI_Wtime();
			double total_time = et - st;

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (total_time == max_time)
					std::cout << "[MPIAlltoallv-2] " << nprocs << " " << n << " "<<  max_time << std::endl;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		delete[] send_buffer;
		delete[] recv_buffer;

	}

}



