/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include <typeinfo>
#include "../src/rbruckv.h"

static int rank, nprocs;
static void run_rbruckv(int loopcount, int nprocs, std::vector<int> bases, int warmup);

int main(int argc, char **argv) {
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    if (argc < 3) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <base-list> " << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);

    std::vector<int> bases;
    for (int i = 2; i < argc; i++)
    	bases.push_back(atoi(argv[i]));

    // warm-up only
//    run_rbruckv(1, nprocs, bases, 1);

    // actual running
    run_rbruckv(loopCount, nprocs, bases, 0);

	MPI_Finalize();
    return 0;
}


static void run_rbruckv(int loopcount, int nprocs, std::vector<int> bases, int warmup) {

	int mpi_errno = MPI_SUCCESS;
	int basecount = bases.size();
	for (int n = 16; n <= 16; n = n * 2) {

		int sendcounts[nprocs]; // the size of data each process send to other process
		memset(sendcounts, 0, nprocs*sizeof(int));
		int sdispls[nprocs];
		int soffset = 0;

		// Uniform random distribution
		srand(time(NULL));
		for (int i=0; i < nprocs; i++) {
//			int random = rand() % 100;
//			sendcounts[i] = (n * random) / 100;
			sendcounts[i] = n;
		}

		// Random shuffling the sendcounts array
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(&sendcounts[0], &sendcounts[nprocs], std::default_random_engine(seed));

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

		// Initial send buffer
		long long* send_buffer = new long long[soffset];
		long long* recv_buffer = new long long[roffset];

		int index = 0;
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < sendcounts[i]; j++)
				send_buffer[index++] = i + rank * 10;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for (int i = 0; i < basecount; i++) {
			int eb = bases[i]+bases[i]/8;
//			for (int b = 1; b < eb; b+=nprocs/8) {
				for (int it=0; it < loopcount; it++) {
					double st = MPI_Wtime();
					mpi_errno = tuna2_algorithm(bases[i], 1, (char*)send_buffer, sendcounts, sdispls,
							MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls,
							MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
					double et = MPI_Wtime();
					double total_time = et - st;

					if (mpi_errno != MPI_SUCCESS)
						std::cout << "twophase_rbruck_alltoallv fail!" <<std::endl;

					// check correctness
					int error = check_errors(recvcounts, recv_buffer, rank, nprocs);

					if (error > 0) {
						std::cout << rank << " " << n << " [Rbruckv] base " << bases[i] << " has errors" << std::endl;
					}

					if (warmup == 0) {
						double max_time = 0;
						MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


						if (total_time == max_time) {
							std::cout << "[Rbruckv] " << nprocs << ", " << n << ", " << 1 << ", " << bases[i] << ", " << max_time << std::endl;
						}
					}
				}
//			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0) {
			int id = 0;
			for (int i = 0; i < nprocs; i++) {
				for (int j = 0; j < 5; j++) {
					std::cout << "recv " << recv_buffer[i*16+j] << std::endl;
				}
			}
		}

//		// MPI_alltoallv
//		for (int it = 0; it < loopcount; it++) {
//			double st = MPI_Wtime();
//			MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
//			double et = MPI_Wtime();
//			double total_time = et - st;
//
//			if (warmup == 0) {
//				double max_time = 0;
//				MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//				if (total_time == max_time)
//					std::cout << "[MPIAlltoallv] " << nprocs << " " << n << " "<<  max_time << std::endl;
//			}
//		}
//		MPI_Barrier(MPI_COMM_WORLD);

		delete[] send_buffer;
		delete[] recv_buffer;
	}

}



