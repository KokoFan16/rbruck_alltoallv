/*
 * ms_benchmark.cpp
 *
 *  Created on: Mar 1, 2024
 *      Author: kokofan
 */


#include "../src/rbruckv.h"

int nprocs, rank;

int main(int argc, char **argv) {

    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");

    int loopcount = 1;

    for (int n = 128; n <= 131072; n = n * 4) {

		long long* send_buffer = new long long[n*nprocs];
		long long* recv_buffer = new long long[n*nprocs];

		for (int p=0; p<n*nprocs; p++) {
			long long value = p/n + rank * 10;
			send_buffer[p] = value;
		}

		for (int l = 1; l <= n; l *= 2) {
			int ncomm = n / l;

			for (int c = 0; c < ncomm; c++){

			}
//			if (rank == 0) {
//				std::cout << n << " " << l << " " << ncomm << std::endl;
//			}
//			for (int it=0; it < loopcount; it++) {
//
//			}
		}

		delete[] send_buffer;
		delete[] recv_buffer;
    }

    MPI_Finalize();
    return 0;
}

