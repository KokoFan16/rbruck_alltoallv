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
int run(int loopcount, int ncores, int csize, int lsize, int warmup);
void simply_merge_alltoallv(int ncores, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
		   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

struct {
    double value;
    int rank;
} local, global;

double other_time = 0, gather_time = 0, ata_time = 0, scatter_time = 0;

int main(int argc, char **argv) {

    if (argc < 4) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <ncores> <csize> <lsize>" << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);
    int ncores = atoi(argv[2]);
    int csize = atoi(argv[3]);
    int lsize = atoi(argv[4]);


    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");


//    run(10, csize, lsize, 1);

    run(loopCount, ncores, csize, lsize, 0);

    MPI_Finalize();
    return 0;
}

int run(int loopcount, int ncores, int csize, int lsize, int warmup) {

	int sendcounts[nprocs], recvcounts[nprocs], sdispls[nprocs], rdispls[nprocs];
	int soffset = 0, roffset = 0, error = 0;

	for (int i=0; i < nprocs; i++) {
		if (rank % ncores == 0) { sendcounts[i] = csize; }
		else { sendcounts[i] = csize; }
	}

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

	int index = 0;
	for (int i = 0; i < nprocs; i++) {
		for (int j = 0; j < sendcounts[i]; j++)
			send_buffer[index++] = i + rank * 10;
	}

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t = 0; t < loopcount; t++) {

    	double start = MPI_Wtime();
    	spreadout_alltoallv((char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
    			(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    	double end = MPI_Wtime();
    	double comm_time = (end - start);

    	local.value = comm_time;
    	local.rank = rank;

    	if (warmup == 0) {
			MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

			if (rank == global.rank) {
				std::cout << "Spread-out " << nprocs << " " << rank << " "
						<< comm_time << " " << csize << " " << lsize << std::endl;
			}
    	}
    }


    for (int t = 0; t < loopcount; t++) {

//    	std::cout << rank << " simply_merge_alltoallv " << std::endl;

    	simply_merge_alltoallv(ncores, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
    			(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    	double total_time = other_time + gather_time + ata_time + scatter_time;

//    	std::cout << rank << " " << total_time << std::endl;

    	if (warmup == 0) {
    		double max_time = 0;
			MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

			if (total_time == max_time) {
				std::cout << "simply_merge " << nprocs << " " << rank << " "
						<< max_time << " " << other_time << " " << gather_time << " " << ata_time
						<< " " << scatter_time << " " << csize << " " << lsize << std::endl;
			}
    	}
    }


	delete[] send_buffer;
	delete[] recv_buffer;

	return 0;
}

void simply_merge_alltoallv(int ncores, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
                     int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	double start = MPI_Wtime();
    int rank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    int group_id, ngroup, local_rank, group_leader;
    MPI_Comm group_comm, leader_comm;

    // Calculate the number of groups
    ngroup = nprocs / ncores;
    group_id = rank / ncores;

    // Split communicator based on color to form groups
    MPI_Comm_split(MPI_COMM_WORLD, group_id, rank, &group_comm);

    // Get local rank and size within the group
    MPI_Comm_rank(group_comm, &local_rank);
    group_leader = group_id * ncores;

    int total_send = sendcounts[0] * nprocs * typesize;
    double end = MPI_Wtime();
    other_time = end - start;

    // Gather data at the group leader
    start = MPI_Wtime();
    char* gathered_data = NULL;
	char* all_to_all_data = NULL;
    if (local_rank == 0) {
    	gathered_data = (char*)malloc(ncores * total_send);
		all_to_all_data = (char*)malloc(ncores * total_send);
    }
    MPI_Gather(sendbuf, total_send, MPI_CHAR, gathered_data, total_send, MPI_CHAR, 0, group_comm);
    end = MPI_Wtime();
    gather_time = end - start;

    start = MPI_Wtime();
	// Create a communicator for group leaders
	MPI_Comm_split(MPI_COMM_WORLD, (local_rank == 0) ? 0 : MPI_UNDEFINED, rank, &leader_comm);

	// Perform all-to-all among group leaders
	if (local_rank == 0) {
		MPI_Alltoall(gathered_data, ncores * ncores * sendcounts[0] * typesize, MPI_CHAR, all_to_all_data, ncores * ncores * sendcounts[0] * typesize, MPI_CHAR, leader_comm);
	}
    end = MPI_Wtime();
    ata_time = end - start;

    start = MPI_Wtime();
	if (local_rank == 0) {
		MPI_Scatter(all_to_all_data, total_send, MPI_CHAR,
				recvbuf, total_send, MPI_CHAR, 0, group_comm);
	}

	// Clean up
	if (local_rank == 0) {
		free(gathered_data);
		free(all_to_all_data);
	}
	MPI_Comm_free(&group_comm);
	if (local_rank == 0) {
		MPI_Comm_free(&leader_comm);
	}
    end = MPI_Wtime();
    scatter_time = end - start;
}









