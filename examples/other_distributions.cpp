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
		int bblock2, int radix2, float dist, float mean, int deviation, float x, int maxValue, int warmup);

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

    if (argc < 11) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <ncores-per-node> "
    			"<bblock1> <radix1> <bblock1> <radix1> <dist> <mean> <deviation> <x> <maxValue>" << std::endl;
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

    // warm-up only
    run_rbruckv(10, ncores, nprocs, bblock1, radix1, bblock2, radix2, dist, mean, deviation, x, maxValue, 1);

    // actual running
    run_rbruckv(loopCount, ncores, nprocs, bblock1, radix1, bblock2, radix2, dist, mean, deviation, x, maxValue, 0);

	MPI_Finalize();
    return 0;
}

static void run_rbruckv(int loopcount, int ncores, int nprocs, int bblock1, int radix1,
		int bblock2, int radix2, float dist, float mean, int deviation, float x, int maxValue, int warmup) {

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
		for(int n = 0; n < nprocs; n++) {
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

//
//		if (rank == 6) {
//			index = 0;
//			for (int i = 0; i < nprocs; i++) {
//				std::cout << i << " " << recvcounts[i] << "[ ";
//				for (int j = 0; j < recvcounts[i]; j++){
//					std::cout << send_buffer[index++] << " ";
//				}
//				std::cout <<  " ]" << std::endl;
//			}
//		}

	delete[] send_buffer;
	delete[] recv_buffer;

}


//void creat_normal_distribution_inputs(int* sendscounts) {
//
//	std::default_random_engine generator;
//	std::normal_distribution<double> distribution(mean, deviation); // set mean and deviation, nprocs/2, nprocs/3
//
//	for (int i = 0; i < nZero; i++) {
//		sendsarray.push_back(0);
//	}
//
//	while(true)
//	{
//		sendsarray.resize(nprocs);
//		int p = int(distribution(generator));
//		if (p >= nZero && p < nprocs) {
//			if (++sendsarray[p] >= maxValue) break;
//		}
//	}
//
//	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//	std::shuffle(&sendsarray[0], &sendsarray[nprocs], std::default_random_engine(seed));
//
//	recvcounts.resize(nprocs);
//	MPI_Alltoall(sendsarray.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
//
//}



//
//void creat_Powerlaw_distribution_inputs(int* &sendscounts, int* &recvcounts) {
//	double x = (double)maxValue;
//
//	sendsarray.resize(nprocs);
//
//	for (int i = 0; i < nZero; i++) {
//		sendsarray.push_back(0);
//	}
//
//	for (int i = nZero; i < nprocs; ++i) {
//		sendsarray[i] = (int)x;
//		x = x * rx; // 0.999
//	}
//
//	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//	std::shuffle(&sendsarray[0], &sendsarray[nprocs], std::default_random_engine(seed));
//
//	recvcounts.resize(nprocs);
//	MPI_Alltoall(sendsarray.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
//}


//#include <stdio.h>
//#include <stdlib.h>
//#include <sys/time.h>
//#include <string.h>
//#include <stdbool.h>
//#include <time.h>
//#include <math.h>
//#include <fstream>
//
//#include <mpi.h>
//
//#include <iostream>
//#include <string>
//#include <sstream>
//#include <vector>
//#include <algorithm>
//#include <numeric>
//#include <random>
//#include <chrono>

// #include "fj_tool/fipp.h"

//int ITE = 50;
//int nprocs, rank;
//int nZero, dist, maxValue;
//float mean, deviation, rx;
//
//int run(int loopcount, std::vector<int>& sendsarray, std::vector<int>& recvcounts);
//
//void creat_normal_distribution_inputs(std::vector<int> &sendsarray, std::vector<int> &recvcounts);
//void creat_Powerlaw_distribution_inputs(std::vector<int> &sendsarray, std::vector<int> &recvcounts);
//
//
////// Main entry
////int main(int argc, char **argv)
////{
////
////
////    if (argc != 7) {
////        printf("Usage: %s <Zero-ratio> <dist> <max_value> <mean> <deviation> <x> \n", argv[0]);
////        exit(-1);
////    }
////
////
////    // MPI Initial
////    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
////        printf("ERROR: MPI_Init error\n");
////    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
////        printf("ERROR: MPI_Comm_size error\n");
////    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
////        printf("ERROR: MPI_Comm_rank error\n");
////
////	// input
////	// std::string filename = argv[1];
////	float p = atof(argv[1]);
////	nZero = p * nprocs;
////	dist = atoi(argv[2]);
////	maxValue = atoi(argv[3]);
////	mean =  atof(argv[4]);
////	deviation =  atof(argv[5]);
////	rx =  atof(argv[6]);
////
////	std::vector<int> sendsarray;
////	std::vector<int> recvcounts;
////
////	if (dist == 1) {
////		creat_normal_distribution_inputs(sendsarray, recvcounts);
////	}
////
////	// Power law distribution
////	if (dist == 2)
////	{
////		creat_Powerlaw_distribution_inputs(sendsarray, recvcounts);
////	}
////
//////	std::cout << "Send " << rank << " " << nprocs << " " << dist << " " << nZero << " " << maxValue << " " << mean << " " << deviation << " " << rx << " ";
//////	for (int i = 0; i < nprocs; i++){
//////		std::cout << sendsarray[i] << " ";
//////	}
//////	std::cout << std::endl;
////
////	// std::vector<int> send_diff, recv_diff;
////	// double send_vari = calculate_variance(sendsarray, rank, send_diff);
////	// double send_stdDev = sqrt(send_vari);
////
////	// double recv_vari = calculate_variance(recvcounts, rank, recv_diff);
////	// double recv_stdDev = sqrt(recv_vari);
////
////    // std::cout <<  "INFO--" << rank << ", " << nzero << ", " << mb << ", " << send_stdDev << ", " << recv_stdDev << ", " << dist << std::endl;
////    // // for (int i = 0; i < nprocs; i++) {
////    // // 	std::cout << rank << ", " << i << ", " << send_diff[i] << ", " << recv_diff[i] << std::endl;
////    // // }
////
//// 	// run(comm_mode, 20, sendsarray, recvcounts, 1); // warm-up
//// 	run(20, sendsarray, recvcounts);
////
////
////    MPI_Finalize();
////    return 0;
////}
//
//
//int run(int loopcount, std::vector<int>& sendsarray, std::vector<int>& recvcounts) {
//
//	int sdispls[nprocs], rdispls[nprocs];
//	long send_tsize = 0, recv_tsize =0;
//	int max_sendn = 0, max_recvn = 0;
//	for (int i = 0; i < nprocs; i++) {
//		sdispls[i] = send_tsize;
//		rdispls[i] = recv_tsize;
//		send_tsize += sendsarray[i];
//		recv_tsize += recvcounts[i];
//		if (sendsarray[i] > max_sendn)
//			max_sendn = sendsarray[i];
//		if (recvcounts[i] > max_recvn)
//			max_recvn = recvcounts[i];
//	}
//
//	char *sendbuf = (char *)malloc(send_tsize);
//    char *recvbuf = (char *)malloc(recv_tsize);
//
//    for (int i = 0; i < send_tsize; i++)
//    	sendbuf[i] = 'a' + rand() % 26;
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//
//    for (int t = 0; t < loopcount; t++) {
//    	// fipp_start();
//    	double start = MPI_Wtime();
//    	MPI_Alltoallv(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
//    	double end = MPI_Wtime();
//    	double comm_time_1 = (end - start);
//
//    	double max_time = 0;
//    	MPI_Allreduce(&comm_time_1, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//    	if (max_time == comm_time_1)
//	    	std::cout << "TIME-0, " << rank << " " << comm_time_1 << " " << send_tsize << " " <<  recv_tsize << " " << nZero << " " << dist << " " << maxValue << " " << mean << " " << deviation << " " << rx << " " << std::endl;
//	    MPI_Barrier(MPI_COMM_WORLD);
//
//    	// else if (comm_mode == 1) {
//    	// 	my_spreadout_nonblocking(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
//    	// }
//    	// else if (comm_mode == 2) {
//    	// 	my_alltoallv_blocking(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
//    	// }
//    	// else if (comm_mode == 3) {
//	    start = MPI_Wtime();
//    	my_sorting_nonblocking(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
//    	end = MPI_Wtime();
//    	double comm_time_2 = (end - start);
//
//    	max_time = 0;
//    	MPI_Allreduce(&comm_time_2, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//    	if (max_time == comm_time_2)
//	    	std::cout << "TIME-3, " << rank << " " << comm_time_2 << " " << send_tsize << " " <<  recv_tsize << " " << nZero << " " << dist << " " << maxValue << " " << mean << " " << deviation << " " << rx << " " << std::endl;
//	    MPI_Barrier(MPI_COMM_WORLD);
//
//    	// }
//    	// else {
//    	// 	std::cout << "Unsupported Mode" << std::endl;
//    	// 	return -1;
//    	// }
//    	// fipp_stop();
//
//    	// if (warmup == 0) {
//
//
//	    // }
//
//	    MPI_Barrier(MPI_COMM_WORLD);
//    }
//
//    free(sendbuf);
//    free(recvbuf);
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//    return 0;
//}
//
//
//
//
//
//
//void creat_normal_distribution_inputs(std::vector<int> &sendsarray, std::vector<int> &recvcounts) {
//
//	std::default_random_engine generator;
//	std::normal_distribution<double> distribution(mean, deviation); // set mean and deviation, nprocs/2, nprocs/3
//
//	for (int i = 0; i < nZero; i++) {
//		sendsarray.push_back(0);
//	}
//
//	while(true)
//	{
//		sendsarray.resize(nprocs);
//		int p = int(distribution(generator));
//		if (p >= nZero && p < nprocs) {
//			if (++sendsarray[p] >= maxValue) break;
//		}
//	}
//
//	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//	std::shuffle(&sendsarray[0], &sendsarray[nprocs], std::default_random_engine(seed));
//
//	recvcounts.resize(nprocs);
//	MPI_Alltoall(sendsarray.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
//
//}
//
//void creat_Powerlaw_distribution_inputs(std::vector<int> &sendsarray, std::vector<int> &recvcounts) {
//	double x = (double)maxValue;
//
//	sendsarray.resize(nprocs);
//
//	for (int i = 0; i < nZero; i++) {
//		sendsarray.push_back(0);
//	}
//
//	for (int i = nZero; i < nprocs; ++i) {
//		sendsarray[i] = (int)x;
//		x = x * rx; // 0.999
//	}
//
//	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//	std::shuffle(&sendsarray[0], &sendsarray[nprocs], std::default_random_engine(seed));
//
//	recvcounts.resize(nprocs);
//	MPI_Alltoall(sendsarray.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
//}





