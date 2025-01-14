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
int run(int loopcount, int warmup);


int main(int argc, char **argv) {

    if (argc < 2) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count>" << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);

    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");

    run(20, 1);
    run(loopCount, 0);

    MPI_Finalize();
    return 0;
}

int run(int loopcount, int warmup) {

	for (int n = 1; n <= 1000000; n = n * 10) {

		char* send_buffer = new char[n*nprocs];
		char* recv_buffer = new char[n*nprocs];

		for (int it=0; it < loopcount; it++) {

			double comm_start = MPI_Wtime();
			MPI_Alltoall(send_buffer, n, MPI_CHAR, recv_buffer, n, MPI_CHAR, MPI_COMM_WORLD);
			double comm_end = MPI_Wtime();
			double total_time = comm_end - comm_start;

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (total_time == max_time)
					std::cout << "[MPIAlltoall] " << nprocs << ", " << n << ", " <<  max_time << std::endl;
			}
		}

		delete[] send_buffer;
		delete[] recv_buffer;
	}

	return 0;
}









