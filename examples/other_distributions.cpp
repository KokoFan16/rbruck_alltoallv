/*
 * radix_r_bruck.cpp
 *
 *  Created on: Jul 09, 2022
 *      Author: kokofan
 */

#include <typeinfo>
#include "../src/rbruckv.h"
#include <random>

static int rank, nprocs;
float mean, deviation, rx;
static void run_rbruckv(int loopcount, int ncores, int nprocs, int bblock1, int radix1,
		int bblock2, int radix2, float dist, float mean, int deviation, float x, int maxValue, int radix0, int warmup);

//void creat_normal_distribution_inputs(int* sendscounts);
//void creat_Powerlaw_distribution_inputs(int* sendscounts);
//

int main(int argc, char **argv) {
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    if (argc < 12) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <ncores-per-node> "
    			"<bblock1> <radix1> <bblock2> <radix2> <dist> <mean> <deviation> <x> <maxValue> <radix0>" << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);
    int ncores = atoi(argv[2]);
    int bblock1 = atoi(argv[3]);
    int radix1 = atoi(argv[4]);
    int bblock2 = atoi(argv[5]);
    int radix2 = atoi(argv[6]);
    int dist = atoi(argv[7]);
    int mean = atoi(argv[8]);
    int deviation = atoi(argv[9]);
    float x = atof(argv[10]);
    int maxValue = atoi(argv[11]);
    int radix0 = atoi(argv[12]);

    // warm-up only
    run_rbruckv(10, ncores, nprocs, bblock1, radix1, bblock2, radix2, dist, mean, deviation, x, maxValue, radix0, 1);

    // actual running
    run_rbruckv(loopCount, ncores, nprocs, bblock1, radix1, bblock2, radix2, dist, mean, deviation, x, maxValue, radix0, 0);

	MPI_Finalize();
    return 0;
}

static void run_rbruckv(int loopcount, int ncores, int nprocs, int bblock1, int radix1,
		int bblock2, int radix2, float dist, float mean, int deviation, float x, int maxValue, int radix0, int warmup) {

	int mpi_errno = MPI_SUCCESS;

	int sendcounts[nprocs], sdispls[nprocs], recvcounts[nprocs], rdispls[nprocs];
	memset(sendcounts, 0, nprocs*sizeof(int));
	int soffset = 0, roffset = 0, index = 0;
	int total_count = 0;

	if (dist == 0) {
		std::random_device rd;
		std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

		std::normal_distribution<double> d(mean, deviation);

		if (rank == 0)
			std::cout << rank << " normal_distribution [";
		for (int n = 0; n < nprocs; n++) {
			double value;
			do {
				value = d(gen);
			} while (value < 0 || value > maxValue);

			sendcounts[n] = (int) value;
			if (rank == 0)
				std::cout << value << ", ";
		}
		if (rank == 0)
			std::cout << "]" << std::endl;
	}

	// Power law distribution
	if (dist == 1) {
		double dx = (double) maxValue;
		if (rank == 0)
			std::cout << rank << " powerLaw_distribution [";
		for(int n = 0; n < nprocs; n++) {
			sendcounts[n] = (int) dx;
			if (rank == 0)
				std::cout << dx << ", ";
			dx = dx * x; // 0.999
		}
		if (rank == 0)
			std::cout << "]" << std::endl;
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
				std::cout << "[MPIAlltoallv] " << nprocs << " " <<  max_time << std::endl;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);



	for (int it=0; it < loopcount; it++) {

		double st = MPI_Wtime();
		mpi_errno = twophase_rbruck_alltoallv(radix0, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		double et = MPI_Wtime();
		double total_time = et - st;

		if (mpi_errno != MPI_SUCCESS)
			std::cout << "twophase_rbruck_alltoallv fail!" <<std::endl;

		// check correctness
		int error = check_errors(recvcounts, recv_buffer, rank, nprocs);

		if (error > 0) {
			std::cout << "[Rbruckv] base " << radix0 << " has errors" << std::endl;
		}

		if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


			if (total_time == max_time) {

				std::cout << "[Rbruckv] " << nprocs << ", " << radix0 << ", " << max_time << std::endl;
			}
		}
	}


	for (int it=0; it < loopcount; it++) {
		double st = MPI_Wtime();
		mpi_errno = TTPL_BT_alltoallv_s1(ncores, radix1, bblock1, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		double et = MPI_Wtime();
		double total_time = et - st;

		if (mpi_errno != MPI_SUCCESS)
			std::cout << "TTPL_BT_alltoallv_s1 fail!" <<std::endl;

		// check correctness
		int error = check_errors(recvcounts, recv_buffer, rank, nprocs);

		if (error > 0) {
			std::cout << "[TTPL_S1] base " << radix1 << " " << bblock1 << " has errors" << std::endl;
//					MPI_Abort(MPI_COMM_WORLD, -1);
		}

		if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

			if (total_time == max_time) {
				double ttime = init_time + findMax_time + rotateIndex_time + alcCopy_time + getBlock_time
						+ prepData_time + excgMeta_time + excgData_time + replace_time + orgData_time
						+ prepSP_time + SP_time;

				std::cout << "[TTPL_S1] " << nprocs << ", " << ncores << ", " << radix1 << ", " << bblock1 << ", " << ttime << std::endl;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	for (int it=0; it < loopcount; it++) {

		double st = MPI_Wtime();
		mpi_errno = TTPL_BT_alltoallv(ncores, radix2, bblock2, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		double et = MPI_Wtime();
		double total_time = et - st;

		if (mpi_errno != MPI_SUCCESS)
			std::cout << "TTPL_BT_alltoallv fail!" <<std::endl;

		// check correctness
		int error = check_errors(recvcounts, recv_buffer, rank, nprocs);

		if (error > 0) {
			std::cout << "[TTPL] base " << radix2 << " " << bblock2 << " has errors" << std::endl;
//					MPI_Abort(MPI_COMM_WORLD, -1);
		}

		if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

			if (total_time == max_time) {
				double ttime = init_time + findMax_time + rotateIndex_time + alcCopy_time + getBlock_time
						+ prepData_time + excgMeta_time + excgData_time + replace_time + orgData_time
						+ prepSP_time + SP_time;

				std::cout << "[TTPL] " << nprocs << ", " << ncores << ", " << bblock2 << ", " << radix2 << ", " << ttime << std::endl;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	delete[] send_buffer;
	delete[] recv_buffer;

}





