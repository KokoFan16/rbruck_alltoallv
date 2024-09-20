/*
 * twophase_tunable_rbruckv.cpp
 *
 *  Created on: Jan 4, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

int twophase_rbruck_alltoallv_om (int r, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	if ( r < 2 ) { return -1; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if ( r > nprocs ) { r = nprocs; }

	int w, nlpow, d;
	int lgc, lc, rd;
	int local_max_count=0, max_send_count=0;
	int sendNcopy[nprocs], rotate_index_array[nprocs];

	MPI_Type_size(sendtype, &typesize);

	w = ceil(log(nprocs) / float(log(r))); // calculate the number of digits when using r-representation
	nlpow = myPow(r, w-1); // maximum send number of elements
	d = (myPow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

	// 1. Find max send count
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	memcpy(sendNcopy, sendcounts, nprocs*sizeof(int));

    // 2. create local index array after rotation
	for (int i = 0; i < nprocs; i++)
		rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;

	// 3. exchange data with log(P) steps
	char* extra_buffer = (char*) malloc(max_send_count*typesize*(nprocs - w - 1));

	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

	int sent_blocks[nlpow];
	int di = 0, spoint = 1, distance = 1, next_distance = distance*r;

	for (int x = 0; x < w; x++) {
		int ze = (x == w - 1)? r - d: r;
		for (int z = 1; z < ze; z++) {

			// 1) get the sent data-blocks
			spoint = z * distance;
			di = 0;

			for (int i = spoint; i < nprocs; i += next_distance) {
				int j_end = (i+distance > nprocs)? nprocs: i+distance;
				for (int j = i; j < j_end; j++) {
					int id = (j + rank) % nprocs;
					sent_blocks[di++] = id;
				}
			}

			int recvrank = (rank + spoint) % nprocs; // receive data from rank - 2^step process
			int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process

			if (di == 1) {
				int send_index = rotate_index_array[sent_blocks[0]];
				MPI_Sendrecv(&sendbuf[sdispls[send_index]*typesize], sendcounts[send_index]*typesize, MPI_CHAR,
						sendrank, 1, &recvbuf[rdispls[sent_blocks[0]]*typesize], recvcounts[sent_blocks[0]]*typesize, MPI_CHAR,
						recvrank, 1, comm, MPI_STATUS_IGNORE);
			}
			else {
				// 2) prepare metadata
				int metadata_send[di];
				int sendCount = 0, offset = 0;
				for (int i = 0; i < di; i++) {
					int send_index = rotate_index_array[sent_blocks[i]];
					metadata_send[i] = sendNcopy[send_index];
					offset += sendNcopy[send_index]*typesize;
				}

				int metadata_recv[di];
				MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0, metadata_recv, di, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

				for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }

				// prepare send and recv buffer
				char* temp_recv_buffer = (char*) malloc(sendCount*typesize);
				char* temp_send_buffer = (char*) malloc(offset);


				offset = 0;
				for (int i = 0; i < di; i++) {
					int send_index = rotate_index_array[sent_blocks[i]];
					int size = sendNcopy[send_index]*typesize;

					if (i % distance == 0) {
						memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], size);
					}
					else {
						int org_id = (sent_blocks[i] - rank + nprocs) % nprocs;
						int logN = log(org_id) / (float)log(r);
						int largest_small_id = myPow(r, logN);
						int dz = org_id / (float) largest_small_id;
						int extra_id = org_id - r - (logN - 1)*(r-1) - dz;

						memcpy(&temp_send_buffer[offset], &extra_buffer[extra_id*max_send_count*typesize], size);
					}

					offset += size;

				}

				// 4) exchange data
				MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);

				// 5) replaces
				offset = 0;
				for (int i = 0; i < di; i++) {
					int send_index = rotate_index_array[sent_blocks[i]];
					int size = metadata_recv[i]*typesize;

					if (i < distance) {
						memcpy(&recvbuf[rdispls[sent_blocks[i]]*typesize], &temp_recv_buffer[offset], size);
					}
					else {
						int org_id = (sent_blocks[i] - rank + nprocs) % nprocs;
						int logN = log(org_id) / (float)log(r);
						int largest_small_id = myPow(r, logN);
						int dz = org_id / (float) largest_small_id;
						int extra_id = org_id - r - (logN - 1)*(r-1) - dz;

						memcpy(&extra_buffer[extra_id*max_send_count*typesize], &temp_recv_buffer[offset], size);
					}
					offset += size;
					sendNcopy[send_index] = metadata_recv[i];
				}
				free(temp_send_buffer);
				free(temp_recv_buffer);
			}
		}
		distance *= r;
		next_distance *= r;
	}

	free(extra_buffer);

	return 0;
}

