/*
 * twophase_tunable_rbruckv.cpp
 *
 *  Created on: Jan 4, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm){

	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if ( r > nprocs ) { r = nprocs; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
	int nlpow = pow(r, w-1); // maximum send number of elements
	int d = (pow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

	// 1. Find max send count
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

	int sendNcopy[nprocs];
	memcpy(sendNcopy, sendcounts, nprocs*sizeof(int));

	// 2. create local index array after rotation
	int rotate_index_array[nprocs];
	for (int i = 0; i < nprocs; i++)
		rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;


	// 3. exchange data with log(P) steps
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*nlpow);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*nlpow);
	int pos_status[nprocs];
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

	int sent_blocks[nlpow];
	int di = 0;
	int spoint = 1, distance = myPow(r, w-1), next_distance = distance*r;
	for (int x = w-1; x > -1; x--) {
		int ze = (x == w - 1)? r - d: r;
		for (int z = ze-1; z > 0; z--) {

			// 1) get the sent data-blocks
			di = 0;
			spoint = z * distance;
			for (int i = spoint; i < nprocs; i += next_distance) {
				for (int j = i; j < (i+distance); j++) {
					if (j > nprocs - 1 ) { break; }
					int id = (j + rank) % nprocs;
					sent_blocks[di++] = id;
				}
			}

			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = sendNcopy[send_index];
				if (pos_status[send_index] == 0)
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], sendNcopy[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], sendNcopy[send_index]*typesize);
				offset += sendNcopy[send_index]*typesize;
			}

			// 3) exchange metadata
			int recvrank = (rank + spoint) % nprocs; // receive data from rank - 2^step process
			int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process

			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0, metadata_recv, di, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++)
				sendCount += metadata_recv[i];

			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);

			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];

				memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				sendNcopy[send_index] = metadata_recv[i];
			}
		}
		distance /= r;
		next_distance /= r;
	}

	// final rotation
	int index = 0;
	for (int i = 0; i < nprocs; i++) {
		if (rank == i) {
			index += recvcounts[i];
			continue;
		}
		int sp = i * max_send_count * typesize;
		for (int j = 0; j < recvcounts[i]; j++) {
			memcpy(&recvbuf[index*typesize], &extra_buffer[sp + j*typesize], typesize);
			index++;
		}
	}

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);

	return 0;
}

