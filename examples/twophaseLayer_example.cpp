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
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <ncores-per-node> <base-list> " << std::endl;
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

	for (int it = 0; it < loopcount; it++) {
		for (int n = 2; n <= 2048; n = n * 2) {

			int sendcounts[nprocs]; // the size of data each process send to other process
			memset(sendcounts, 0, nprocs*sizeof(int));
			int sdispls[nprocs];
			int soffset = 0;

			// Uniform random distribution
			srand(time(NULL));
			for (int i=0; i < nprocs; i++) {
				int random = rand() % 100;
				sendcounts[i] = (n * random) / 100;
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

			// MPI_alltoallv
//			for (int it = 0; it < loopcount; it++) {
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
//			}

			MPI_Barrier(MPI_COMM_WORLD);


			// MPICH_intra_scattered
			for (int b = 1; b <= nprocs; b*=2){
//				for (int it = 0; it < loopcount; it++) {
					double st = MPI_Wtime();
					mpi_errno = MPICH_intra_scattered(b, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
					double et = MPI_Wtime();
					double total_time = et - st;

					if (mpi_errno != MPI_SUCCESS)
						std::cout << "MPICH_intra_scattered fail!" <<std::endl;

					// check correctness
					int error = check_errors(recvcounts, recv_buffer, rank, nprocs);
					if (error > 0) {
						std::cout << "[MPICH_scattered] " << n << " " << b << " has errors" << std::endl;
	//					MPI_Abort(MPI_COMM_WORLD, -1);
					}

					if (warmup == 0) {
						double max_time = 0;
						MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
						if (total_time == max_time)
							std::cout << "[MPICH_scattered] " << nprocs << " " << n << " " << b << " " << max_time << std::endl;
					}
//				}
			}

			MPI_Barrier(MPI_COMM_WORLD);

	//		 ompi_alltoallv_intra_basic_linear
//			for (int it = 0; it < loopcount; it++) {
				st = MPI_Wtime();
				mpi_errno = ompi_alltoallv_intra_basic_linear((char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
				et = MPI_Wtime();
				total_time = et - st;

				if (mpi_errno != MPI_SUCCESS)
					std::cout << "ompi_alltoallv_intra_basic_linear fail!" <<std::endl;

				// check correctness
				int error = check_errors(recvcounts, recv_buffer, rank, nprocs);
				if (error > 0) {
					std::cout << "[OMPI_linear] " << n << " has errors" << std::endl;
	//				MPI_Abort(MPI_COMM_WORLD, -1);
				}

				if (warmup == 0) {
					double max_time = 0;
					MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
					if (total_time == max_time)
						std::cout << "[OMPI_linear] " << nprocs << " " << n << " "<<  max_time << std::endl;
				}
//			}

			MPI_Barrier(MPI_COMM_WORLD);


	//		 ompi_alltoallv_intra_basic_linear
//			for (int it = 0; it < loopcount; it++) {
				st = MPI_Wtime();
				mpi_errno = ompi_alltoallv_intra_pairwise((char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
				et = MPI_Wtime();
				total_time = et - st;

				if (mpi_errno != MPI_SUCCESS)
					std::cout << "ompi_alltoallv_intra_pairwise fail!" <<std::endl;

				// check correctness
				error = check_errors(recvcounts, recv_buffer, rank, nprocs);
				if (error > 0) {
					std::cout << "[OMPI_pairwise] " << n << " has errors" << std::endl;
	//				MPI_Abort(MPI_COMM_WORLD, -1);
				}

				if (warmup == 0) {
					double max_time = 0;
					MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
					if (total_time == max_time)
						std::cout << "[OMPI_pairwise] " << nprocs << " " << n << " "<<  max_time << std::endl;
//				}
			}

			MPI_Barrier(MPI_COMM_WORLD);

			// exclisive_or_alltoallv (limit: P is powers of 2)
//			for (int it = 0; it < loopcount; it++) {
				st = MPI_Wtime();
				mpi_errno = exclisive_or_alltoallv((char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
				et = MPI_Wtime();
				total_time = et - st;

				if (mpi_errno != MPI_SUCCESS) { std::cout << "exclisive_or_alltoallv fail!" <<std::endl; }

				// check correctness
				error = check_errors(recvcounts, recv_buffer, rank, nprocs);
				if (error > 0) {
					std::cout << "[ExcOr] " << n << " has errors" << std::endl;
	//				MPI_Abort(MPI_COMM_WORLD, -1);
				}

				if (warmup == 0) {
					double max_time = 0;
					MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
					if (total_time == max_time)
						std::cout << "[ExcOr] " << nprocs << " " << n << " " << max_time << std::endl;
//				}
			}

			MPI_Barrier(MPI_COMM_WORLD);

	//		if (rank == 1) {
	//			index = 0;
	//			for (int i = 0; i < nprocs; i++) {
	//				for (int j = 0; j < recvcounts[i]; j++){
	//					std::cout << recv_buffer[index++] << std::endl;
	//				}
	//			}
	//		}

			delete[] send_buffer;
			delete[] recv_buffer;
		}
	}

}



