/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include <typeinfo>
#include "../src/rbruckv.h"

static int rank, nprocs;
static void run_rbruckv(int loopcount, int ncores, int nprocs, std::vector<int> bases, int warmup);

int main(int argc, char **argv) {
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    if (argc < 4) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <ncores-per-node> <sblock> <base-list> " << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);
    int ncores = atoi(argv[2]);

    std::vector<int> bases;
    for (int i = 3; i < argc; i++)
    	bases.push_back(atoi(argv[i]));

//     warm-up only
//    run_rbruckv(5, ncores, nprocs, bases, 1);

    // actual running
    run_rbruckv(loopCount, ncores, nprocs, bases, 0);

	MPI_Finalize();
    return 0;
}

static void run_rbruckv(int loopcount, int ncores, int nprocs, std::vector<int> bases, int warmup) {

	int mpi_errno = MPI_SUCCESS;
	int basecount = bases.size();

	for (int n = 2; n <= 2048; n = n * 2) {

		int sendcounts[nprocs], sdispls[nprocs], recvcounts[nprocs], rdispls[nprocs];
		memset(sendcounts, 0, nprocs*sizeof(int));
		int soffset = 0, roffset = 0, index = 0;
		unsigned seed;
		int max_bblock = nprocs;

		// Uniform random distribution
		srand(time(NULL));
		for (int i=0; i < nprocs; i++) {
//			sendcounts[i] = rank + 1;
			int random = rand() % 100;
			sendcounts[i] = (n * random) / 100;
		}

		// Random shuffling the sendcounts array
		seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(&sendcounts[0], &sendcounts[nprocs], std::default_random_engine(seed));

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
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < sendcounts[i]; j++)
				send_buffer[index++] = i + rank * 10;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for (int i = 0; i < basecount; i++) {
			for (int b = 1; b <= nprocs; b *= 2) {
				for (int it=0; it < loopcount; it++) {

					double st = MPI_Wtime();
					mpi_errno = TTPL_BT_alltoallv(ncores, bases[i], b, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
					double et = MPI_Wtime();
					double total_time = et - st;

					if (mpi_errno != MPI_SUCCESS)
						std::cout << "TTPL_BT_alltoallv fail!" <<std::endl;

					// check correctness
					int error = check_errors(recvcounts, recv_buffer, rank, nprocs);

					if (error > 0) {
						std::cout << "[TTPL] base " << bases[i] << " has errors" << std::endl;
	//					MPI_Abort(MPI_COMM_WORLD, -1);
					}

					if (warmup == 0) {
						double max_time = 0;
						MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

						if (total_time == max_time) {
							double ttime = init_time + findMax_time + rotateIndex_time + alcCopy_time + getBlock_time
									+ prepData_time + excgMeta_time + excgData_time + replace_time + orgData_time
									+ prepSP_time + SP_time;

							std::cout << "[TTPL] " << nprocs << ", " << n << ", " << b << ", " << bases[i] << ", " << ttime <<
							" [" << init_time << ", " << findMax_time << ", " << rotateIndex_time << ", " <<
							alcCopy_time << ", " << getBlock_time << ", " << prepData_time << ", " << excgMeta_time
							<< ", " << excgData_time << ", " << replace_time << ", " << orgData_time << ", " << prepSP_time << ", "
							<< SP_time << "] " << std::endl;
						}
					}
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);


		if (nprocs == 8192) {
			if (n < 8 ) { max_bblock = 2048; }
			else if (n < 512 ) { max_bblock = 1024; }
			else { max_bblock = 512; }
		}
		for (int i = 0; i < basecount; i++) {
			for (int b = 1; b <= max_bblock; b *= 2) {
				for (int it=0; it < loopcount; it++) {

					double st = MPI_Wtime();
					mpi_errno = TTPL_BT_alltoallv_s1(ncores, bases[i], b, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
					double et = MPI_Wtime();
					double total_time = et - st;

					if (mpi_errno != MPI_SUCCESS)
						std::cout << "TTPL_BT_alltoallv_s1 fail!" <<std::endl;

					// check correctness
					int error = check_errors(recvcounts, recv_buffer, rank, nprocs);

					if (error > 0) {
						std::cout << "[TTPL_S1] base " << bases[i] << " has errors" << std::endl;
	//					MPI_Abort(MPI_COMM_WORLD, -1);
					}

					if (warmup == 0) {
						double max_time = 0;
						MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

						if (total_time == max_time) {
							double ttime = init_time + findMax_time + rotateIndex_time + alcCopy_time + getBlock_time
									+ prepData_time + excgMeta_time + excgData_time + replace_time + orgData_time
									+ prepSP_time + SP_time;

							std::cout << "[TTPL_S1] " << nprocs << ", " << n << ", " << b << ", " << bases[i] << ", " << ttime <<
							" [" << init_time << ", " << findMax_time << ", " << rotateIndex_time << ", " <<
							alcCopy_time << ", " << getBlock_time << ", " << prepData_time << ", " << excgMeta_time
							<< ", " << excgData_time << ", " << replace_time << ", " << SP_time << "] " << std::endl;
						}
					}
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// MPI_alltoallv
		for (int it = 0; it < loopcount; it++) {
			double st = MPI_Wtime();
			mpi_errno = MPI_Alltoallv(send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double et = MPI_Wtime();
			double total_time = et - st;

			if (mpi_errno != MPI_SUCCESS)
				std::cout << "MPI_Alltoallv fail!" <<std::endl;


			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (total_time == max_time)
					std::cout << "[MPIAlltoallv] " << nprocs << " " << n << " "<<  max_time << std::endl;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
//
//		if (rank == 6) {
//			index = 0;
//			for (int i = 0; i < nprocs; i++) {
//				std::cout << i << " " <<  recvcounts[i] << "[ ";
//				for (int j = 0; j < recvcounts[i]; j++){
//					std::cout << recv_buffer[index++] << " ";
//				}
//				std::cout <<  " ]" << std::endl;
//			}
//		}

		delete[] send_buffer;
		delete[] recv_buffer;
	}

}



