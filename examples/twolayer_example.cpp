/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include <typeinfo>
#include "../src/rbruckv.h"

static int rank, nprocs;
static void run_twolayer(int loopcount, int ncores, int nprocs, std::vector<int> bases, int warmup);

int main(int argc, char **argv) {
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    if (argc < 4) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <ncores-per-node> <base-list> " << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);
    int ncores = atoi(argv[2]);

    std::vector<int> bases;
    for (int i = 3; i < argc; i++)
    	bases.push_back(atoi(argv[i]));

    // warm-up only
//    run_twolayer(5, ncores, nprocs, bases, 1);

    // actual running
    run_twolayer(loopCount, ncores, nprocs, bases, 0);

	MPI_Finalize();
    return 0;
}


static void run_twolayer(int loopcount, int ncores, int nprocs, std::vector<int> bases, int warmup) {

	int basecount = bases.size();
	for (int n = 2; n <= 2; n = n * 2) {


		long long* send_buffer = new long long[n*nprocs];
		long long* recv_buffer = new long long[n*nprocs];

		for (int it=0; it < loopcount; it++) {

			double comm_start = MPI_Wtime();
			MPI_Alltoall((char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double comm_end = MPI_Wtime();
			double total_time = comm_end - comm_start;

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (total_time == max_time)
					std::cout << "[MPIAlltoall] " << nprocs << ", " << n << ", " <<  max_time << std::endl;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);


		for (int i = 0; i < basecount; i++) {
			for (int it=0; it < loopcount; it++) {

				for (int p=0; p<n*nprocs; p++) {
					long long value = p/n + rank * 10;
					send_buffer[p] = value;
				}

				double st = MPI_Wtime();
				uniform_spreadout_twolayer(ncores, bases[i], (char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
				double et = MPI_Wtime();
				double total_time = et - st;

					// check if correct
					int error = 0;
					for (int d = 0; d < n*nprocs; d++) {
						if ( (recv_buffer[d] % 10) != (rank % 10) ) error++;
					}
					if (rank == 0 && error > 0)
						std::cout << "[SOR_alltoall] " << bases[i] << " has errors" << std::endl;


					if (warmup == 0) {
						double max_time = 0;
						MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

						if (total_time == max_time)
							std::cout << "[SOR_alltoall] " << nprocs << ", " << n << ", " << bases[i] << ", " << max_time << std::endl;
					}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for (int i = 0; i < basecount; i++) {
			for (int it=0; it < loopcount; it++) {

				for (int p=0; p<n*nprocs; p++) {
					long long value = p/n + rank * 10;
					send_buffer[p] = value;
				}

				double st = MPI_Wtime();
				uniform_inverse_isplit_r_bruck(ncores, bases[i], bases[i], (char*)send_buffer, n, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, n, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
				double et = MPI_Wtime();
				double total_time = et - st;

					// check if correct
					int error = 0;
					for (int d = 0; d < n*nprocs; d++) {
						if ( (recv_buffer[d] % 10) != (rank % 10) ) error++;
					}
					if (rank == 0 && error > 0)
						std::cout << "[BR_alltoall] " << bases[i] << " has errors" << std::endl;


					if (warmup == 0) {
						double max_time = 0;
						MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

						if (total_time == max_time)
							std::cout << "[BR_alltoall] " << nprocs << ", " << n << ", " << bases[i] << ", " << max_time << std::endl;
					}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		delete[] send_buffer;
		delete[] recv_buffer;
	}

}



