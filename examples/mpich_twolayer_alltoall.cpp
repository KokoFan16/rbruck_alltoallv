/*
 * mpich_twolayer_alltoallv.cpp
 *
 *  Created on: Feb 15, 2024
 *      Author: kokofan
 */


#include "../src/rbruckv.h"

int main(int argc, char *argv[]) {

    if (argc < 2) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << "<ncores-per-node>" << std::endl;
    	return -1;
    }

    int ncores = atoi(argv[1]);

    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = 1;

	long long* send_buffer = new long long[n*world_size];
	long long* recv_buffer = new long long[n*world_size];

	for (int p=0; p<n*world_size; p++) {
		long long value = p/n + world_rank * 10;
		send_buffer[p] = value;
	}

	memset(recv_buffer, 0, n*world_size*sizeof(long long));

    // Assuming an even number of processes, split into two groups
    int color = world_rank/ncores; // Group identifier
    MPI_Comm split_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &split_comm);

    // Find out my rank and size in the new communicator
    int local_rank, local_size;
    MPI_Comm_rank(split_comm, &local_rank);
    MPI_Comm_size(split_comm, &local_size);


    MPI_Alltoall(send_buffer, n, MPI_LONG_LONG, recv_buffer, n, MPI_LONG_LONG, split_comm);

	int local_leader = 0; // Assuming rank 0 in each split_comm as the leader
	int remote_leader = (color == 0) ? (world_size / 2) : 0; // The remote leader's rank in MPI_COMM_WORLD
	MPI_Comm intercomm;


	int unit = ncores*sizeof(long long);
    // Creating an intercommunicator
    MPI_Intercomm_create(split_comm, local_leader, MPI_COMM_WORLD, remote_leader, 0, &intercomm);

    int inter_rank, inter_size;
    MPI_Comm_rank(intercomm, &inter_rank);
    MPI_Comm_size(intercomm, &inter_size);

    char* recvaddr = (char *) recv_buffer + n * unit;

//    std::cout << world_rank << ", " << inter_rank << ", " << color << std::endl;

    MPI_Alltoall(send_buffer, n, MPI_LONG_LONG, recvaddr, n, MPI_LONG_LONG, intercomm);

	if (world_rank == 1) {
		for (int i = 0; i < world_size; i++) {
			std::cout << recv_buffer[i] << std::endl;
		}
	}


	MPI_Comm_free(&split_comm);
    MPI_Comm_free(&intercomm);

	delete[] send_buffer;
	delete[] recv_buffer;

    MPI_Finalize();
    return 0;
}



//#include <mpi.h>
//#include <stdio.h>
//#include <stdlib.h>
//
//int main(int argc, char** argv) {
//    MPI_Init(&argc, &argv);
//
//    int world_rank, world_size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//
//    // Manually categorize processes into two "nodes" for simplicity
//    int node_id = world_rank < (world_size / 2) ? 0 : 1;
//
//    // Create intra-node communicators
//    MPI_Comm intra_node_comm;
//    MPI_Comm_split(MPI_COMM_WORLD, node_id, world_rank, &intra_node_comm);
//
//    // Now, perform intra-node all-to-all communication
//    // Prepare data for all-to-all (simplified for demonstration)
//    int send_data = world_rank; // Each process sends its rank
//    int recv_data; // Each process receives one item in this simple case
//
//    MPI_Alltoall(&send_data, 1, MPI_INT, &recv_data, 1, MPI_INT, intra_node_comm);
//
//    printf("Process %d on node %d received %d in intra-node all-to-all\n", world_rank, node_id, recv_data);
//
//    // Prepare for inter-node communication by selecting a representative (e.g., rank 0 in each node)
//    int is_representative = (world_rank == 0 || world_rank == world_size / 2) ? 1 : 0;
//
//    // Create inter-node communicator
//    MPI_Comm inter_node_comm;
//    MPI_Comm_split(MPI_COMM_WORLD, is_representative, world_rank, &inter_node_comm);
//
//    // Perform inter-node all-to-all if this process is a representative
//    if (is_representative) {
//        int inter_send_data = node_id; // Send node_id as data
//        int inter_recv_data; // Receive one item
//
//        // Assuming 2 nodes for simplicity; adjust for more nodes
//        MPI_Alltoall(&inter_send_data, 1, MPI_INT, &inter_recv_data, 1, MPI_INT, inter_node_comm);
//        printf("Node representative %d received %d in inter-node all-to-all\n", world_rank, inter_recv_data);
//    }
//
//    // Cleanup
//    MPI_Comm_free(&intra_node_comm);
//    if (is_representative) {
//        MPI_Comm_free(&inter_node_comm);
//    }
//
//    MPI_Finalize();
//    return 0;
//}


