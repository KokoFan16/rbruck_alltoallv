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
void readinputs(int rank, std::string filename, std::vector<int> &sendsarray, std::vector<int> &recvcounts);
std::string getNItem(std::string str, int k, std::string delim, int count);
int run(int loopcount, int ncores, std::vector<int> bases, int warmup, int b, std::vector<int>& sendsarray, std::vector<int>& recvcounts);

int main(int argc, char **argv) {

    if (argc < 6) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <inputFile> <loop-count> <ncores-per-node> <bblock> <base-list> " << std::endl;
    	return -1;
    }

    std::string filename = argv[1];
    int loopCount = atoi(argv[2]);
    int ncores = atoi(argv[3]);
    int bblock = atoi(argv[4]);

    std::vector<int> bases;
    for (int i = 5; i < argc; i++)
    	bases.push_back(atoi(argv[i]));

    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");

	std::vector<int> sendsarray;
	std::vector<int> recvcounts;

    readinputs(rank, filename, sendsarray, recvcounts);

//    run(10, ncores, bases, 1, sendsarray, recvcounts);
//
    run(loopCount, ncores, bases, 0, bblock, sendsarray, recvcounts);
//

    MPI_Finalize();
    return 0;
}

int run(int loopcount, int ncores, std::vector<int> bases, int warmup, int b, std::vector<int>& sendsarray, std::vector<int>& recvcounts) {

	int basecount = bases.size();
	int sdispls[nprocs], rdispls[nprocs];
	long send_tsize = 0, recv_tsize =0;
	int max_sendn = 0, max_recvn = 0;

	for (int i = 0; i < nprocs; i++) {
		sdispls[i] = send_tsize;
		rdispls[i] = recv_tsize;
		send_tsize += sendsarray[i];
		recv_tsize += recvcounts[i];
		if (sendsarray[i] > max_sendn)
			max_sendn = sendsarray[i];
		if (recvcounts[i] > max_recvn)
			max_recvn = recvcounts[i];
	}

	char *sendbuf = (char *)malloc(send_tsize);
    char *recvbuf = (char *)malloc(recv_tsize);

    for (int i = 0; i < send_tsize; i++)
    	sendbuf[i] = 'a' + rand() % 26;

    MPI_Barrier(MPI_COMM_WORLD);


    for (int t = 0; t < loopcount; t++) {

    	double start = MPI_Wtime();
    	MPI_Alltoallv(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
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

			double start = MPI_Wtime();
			TTPL_BT_alltoallv(ncores, bases[i], b, (char*)sendbuf, sendsarray.data(), sdispls, MPI_CHAR, (char*)recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
			double end = MPI_Wtime();
			double comm_time = (end - start);

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (max_time == comm_time)
					std::cout << "LTRNA_S1, " << nprocs << " " << b << " " << bases[i] << ""  << comm_time << " " << send_tsize << " " <<  recv_tsize << std::endl;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

    free(sendbuf);
    free(recvbuf);

	return 0;
}

void readinputs(int rank, std::string filename, std::vector<int> &sendsarray, std::vector<int> &recvcounts) {

	std::ifstream file(filename);
	std::string str;
	int count = 0;

    while (std::getline(file, str)) {

    	std::string sitem = getNItem(str, count, " ", 0);
//    	std::cout << str << std::endl;

    	for (int i = 0; i < 512; i++) {
    		std::string item = getNItem(str, i, " ", 0);
    		recvcounts.push_back(stol(item));
    	}

    	std::string item = getNItem(str, rank, " ", 0);
    	recvcounts.push_back(stol(item));

    	if (rank == count) {
    		std::stringstream ss(str);
    		std::string number;
    		while (ss >> number) sendsarray.push_back(stol(number));
    	}
        count++;
    }
}

std::string getNItem(std::string str, int k, std::string delim, int count) {
	int p = str.find(' ');
	std::string item = str.substr(0, str.find(' '));

	if (count == k) return item;

	str = str.substr(p + delim.length());
	count += 1;

	return getNItem(str, k, delim, count);
}


