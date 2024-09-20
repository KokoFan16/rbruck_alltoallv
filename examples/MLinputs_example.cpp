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
int run(int loopcount, std::vector<int> bases, int warmup, int sendcounts[], int recvcounts[]);
void read_file(int sendcounts[], int recvcounts[], const std::string &filename);

int main(int argc, char **argv) {

    if (argc < 4) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <inputFile> <loop-count> <base-list> " << std::endl;
    	return -1;
    }

    std::string filename = argv[1];
    int loopCount = atoi(argv[2]);
    std::vector<int> bases;
    for (int i = 3; i < argc; i++)
    	bases.push_back(atoi(argv[i]));

    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");


    int sendcounts[nprocs], recvcounts[nprocs];
    read_file(sendcounts, recvcounts, filename);

//    run(10, ncores, bases, 1, sendsarray, recvcounts);
//
    run(loopCount, bases, 0, sendcounts, recvcounts);
//

    MPI_Finalize();
    return 0;
}

int run(int loopcount, std::vector<int> bases, int warmup, int sendcounts[], int recvcounts[]) {

	int basecount = bases.size();
	int sdispls[nprocs], rdispls[nprocs];
	long send_tsize = 0, recv_tsize =0;

	for (int i = 0; i < nprocs; i++) {
		sdispls[i] = send_tsize;
		rdispls[i] = recv_tsize;
		send_tsize += sendcounts[i];
		recv_tsize += recvcounts[i];
	}

	char *sendbuf = (char *)malloc(send_tsize);
    char *recvbuf = (char *)malloc(recv_tsize);

    for (int i = 0; i < send_tsize; i++)
    	sendbuf[i] = 'a' + rand() % 26;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < loopcount; t++) {

    	double start = MPI_Wtime();
    	MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_CHAR, recvbuf, recvcounts, rdispls, MPI_CHAR, MPI_COMM_WORLD);
    	double end = MPI_Wtime();
    	double comm_time = (end - start);

    	if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (max_time == comm_time)
				std::cout << "MPI_Alltoallv, " << nprocs << " " << comm_time << " " << send_tsize << " " <<  recv_tsize << std::endl;
    	}
    }

	MPI_Barrier(MPI_COMM_WORLD);


	for (int i = 0; i < basecount; i++) {
		for (int it=0; it < loopcount; it++) {

			if (rank == 0) {
				std::cout << i << " " << it << " twophase_rbruck_alltoallv" << std::endl;
			}
			double start = MPI_Wtime();
			twophase_rbruck_alltoallv(bases[i], (char*)sendbuf, sendcounts, sdispls, MPI_CHAR, (char*)recvbuf, recvcounts, rdispls, MPI_CHAR, MPI_COMM_WORLD);
			double end = MPI_Wtime();
			double comm_time = (end - start);

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (max_time == comm_time)
					std::cout << "TUNA, " << nprocs << " " << bases[i] << " " << comm_time << " " << send_tsize << " " <<  recv_tsize << std::endl;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

    free(sendbuf);
    free(recvbuf);

	return 0;
}



void read_file(int sendcounts[], int recvcounts[], const std::string &filename) {
    std::ifstream file(filename);
    std::string line;
    int matrix[nprocs][nprocs];

    // Read the matrix from the file
    if (file.is_open()) {
        int i = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            for (int j = 0; j < nprocs; j++) {
                ss >> matrix[i][j];
            }
            i++;
        }
    }

    // Extract the row corresponding to the process rank
    for (int j = 0; j < nprocs; j++) {
    	sendcounts[j] = matrix[rank][j];
    }

    // Extract the column corresponding to the process rank
    for (int i = 0; i < nprocs; i++) {
    	recvcounts[i] = matrix[i][rank];
    }
}


