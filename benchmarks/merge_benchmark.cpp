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
void merge_alltoallv(int ncores, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
		   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

struct {
    double value;
    int rank;
} local, global;

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
		if (rank % ncores == 0) { sendcounts[i] = lsize; }
		else { sendcounts[i] = csize; }
	}

	// Initial send offset array
	for (int i = 0; i < nprocs; ++i) {
		sdispls[i] = soffset;
		soffset += sendcounts[i];
	}

	// Initial receive counts and offset array
	MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

//	if (rank == 4) {
//		for (int i = 0; i < nprocs; i++) {
//			std::cout << recvcounts[i] << std::endl;
//		}
//	}

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

	MPI_Barrier(MPI_COMM_WORLD);

	if (csize != lsize) {
		for (int t = 0; t < loopcount; t++) {
			double start = MPI_Wtime();
			merge_alltoallv(ncores, (char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG,
					(char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
			double end = MPI_Wtime();
			double comm_time = (end - start);

//			// check correctness
//			error = check_errors(recvcounts, recv_buffer, rank, nprocs);
//			if (error > 0) {
//				std::cout << "[Merge] has errors for " << csize << " " << lsize << std::endl;
//			}

			local.value = comm_time;
			local.rank = rank;

			if (warmup == 0) {
				double max_time = 0;
				MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

				if (rank == global.rank) {
					std::cout << "Merge " << nprocs << " " << rank << " "
							<< comm_time << " " << csize << " " << lsize << std::endl;
				}
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	delete[] send_buffer;
	delete[] recv_buffer;

	return 0;
}


void merge_alltoallv(int ncores, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf,
                     int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

    int rank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    int color, ngroup, local_rank, group_size;
    MPI_Comm group_comm;

    // Calculate the number of groups
    ngroup = nprocs / ncores;
    color = rank / ncores;

    // Split communicator based on color to form groups
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &group_comm);

    // Get local rank and size within the group
    MPI_Comm_rank(group_comm, &local_rank);
    MPI_Comm_size(group_comm, &group_size);

    int total_recv = recvcounts[0] * ngroup + recvcounts[1] * (nprocs - ngroup);

    char *gather_buffer = NULL;
    int group_data = recvcounts[1] * nprocs;

    // gather data into rank 1 from all ranks except rank 0
    MPI_Request *requests = NULL;
    if (local_rank == 1) {
        // Allocate gather_buffer for receiving data from all other ranks in the group
        gather_buffer = (char *)malloc(total_recv * (ncores - 1) * typesize);

        // Allocate request array for non-blocking receives
        requests = (MPI_Request *)malloc((ncores - 1) * sizeof(MPI_Request));

        for (int i = 1; i < ncores; i++) {
            // Initiate non-blocking receive for each process's data
            MPI_Irecv(gather_buffer + (i - 1) * group_data * typesize,
                      group_data, recvtype, i, 0, group_comm, &requests[i - 1]);
        }
    }


    // Non-blocking send from each process (except local rank 0) to local rank 1
    if (local_rank != 0) {
        MPI_Request send_request;
        MPI_Isend(sendbuf, group_data, sendtype, 1, 0, group_comm, &send_request);

        // Wait for the send to complete
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    }


    char *reorder_buffer = NULL;
    // Wait for all non-blocking receives to complete at local rank 1
    if (local_rank == 1) {
        MPI_Waitall(ncores - 1, requests, MPI_STATUSES_IGNORE);
        free(requests);

        // Rearrange the data in gather_buffer
        reorder_buffer = (char *)malloc(total_recv * (ncores - 1) * typesize);
        int unit_copy = recvcounts[1] * typesize;
		for (int i = 0; i < nprocs; i++) {
			for (int j = 0; j < (ncores - 1); j++) {
			 memcpy(&reorder_buffer[(i* (ncores - 1) + j) * unit_copy],
					 &gather_buffer[ (j * nprocs + i) * unit_copy], unit_copy);
			}
		}

		memset(gather_buffer, 0, (ncores - 1) * group_data * typesize);
    }


    int num_procs_ata = 2*ngroup;
    int new_sendcounts[num_procs_ata];
    int new_recvcounts[num_procs_ata];
    int new_sdispls[num_procs_ata];
    int new_rdispls[num_procs_ata];

    for (int i = 0; i < num_procs_ata; i++) {

    	int index = (i / 2) * ncores + (i % 2);
    	int unit = (local_rank == 1) ? (ncores - 1) : 1;

    	if (i % 2 == 0) {
    		new_sendcounts[i] = sendcounts[index] * unit;
    		new_recvcounts[i] = recvcounts[index] * unit;
    	}
    	else {
    		new_sendcounts[i] = sendcounts[index] * (ncores - 1) * unit;
    		new_recvcounts[i] = recvcounts[index] * (ncores - 1) * unit;
    	}
    }

    int soffset = 0, roffset = 0;
	for (int i = 0; i < num_procs_ata; i++) {
		new_sdispls[i] = soffset;
		soffset += new_sendcounts[i];

		new_rdispls[i] = roffset;
		roffset += new_recvcounts[i];
	}

	// all-to-all
	MPI_Request* req = (MPI_Request*)malloc(2*num_procs_ata*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*num_procs_ata*sizeof(MPI_Status));
	int nreq = 0;

	for (int i = 0; i < num_procs_ata; i++) {
		int curt_rank = color * 2 + local_rank;
		int src = (curt_rank + i) % num_procs_ata; // avoid always to reach first master node
		int index = src / 2 * ncores + (src % 2);

		if (local_rank == 0) {
			MPI_Irecv(recvbuf + new_rdispls[src] * typesize, new_recvcounts[src],
								recvtype, index, 0, comm, &req[nreq++]);
		}
		if (local_rank == 1) {
			MPI_Irecv(gather_buffer + new_rdispls[src] * typesize, new_recvcounts[src],
								recvtype, index, 0, comm, &req[nreq++]);
		}
	}

	for (int i = 0; i < num_procs_ata; i++) {
		int curt_rank = color * 2 + local_rank;
		int dst = (curt_rank - i + num_procs_ata) % num_procs_ata;
		int index = dst / 2 * ncores + (dst % 2);

		if (local_rank == 0) {
			MPI_Isend(sendbuf + new_sdispls[dst] * typesize, new_sendcounts[dst],
								recvtype, index, 0, comm, &req[nreq++]);
		}
		if (local_rank == 1) {
			MPI_Isend(reorder_buffer + new_sdispls[dst] * typesize, new_sendcounts[dst],
								recvtype, index, 0, comm, &req[nreq++]);
		}
	}
	MPI_Waitall(nreq, req, stat);
	free(req);
	free(stat);

	// orange data
	if (local_rank == 1) {
		roffset = 0;
		for (int i = 0; i < (ncores - 1); i++) {
			for (int j = 0; j < num_procs_ata; j++) {
				int unit_copy = new_recvcounts[j]/(ncores - 1) * typesize;
				memcpy(&reorder_buffer [roffset],
						 &gather_buffer[ new_rdispls[j] * typesize + i * unit_copy ], unit_copy);
				roffset += unit_copy;
			}
		}
	}

//    if (local_rank != 0) {
//        MPI_Recv(recvbuf, total_recv, recvtype, 1, 0, group_comm, MPI_STATUS_IGNORE);
//    }
//
//    if (local_rank == 1) {
//        MPI_Request send_request;
//        for (int i = 1; i < ncores - 1; i++) {
//            MPI_Isend(gather_buffer + (i - 1) * total_recv * typesize, total_recv, sendtype, i, 0, group_comm, &send_request);
//            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
//        }
//    }


    // Clean up and free allocated memory
    if (local_rank == 1) {
        free(reorder_buffer);
        free(gather_buffer);
    }

    MPI_Comm_free(&group_comm);
}









