/*
 * twophase_tunable_rbruckv.cpp
 *
 *  Created on: Jan 4, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

//double init_time = 0, findMax_time = 0, rotateIndex_time = 0, alcCopy_time = 0,
//getBlock_time = 0, prepData_time = 0, excgMeta_time = 0, excgData_time = 0, replace_time = 0;

int twophase_rbruck_alltoallv(int r, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm){

//	double st = MPI_Wtime();
	if ( r < 2 ) { return -1; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if ( r > nprocs ) { r = nprocs; }

	int w, nlpow, d;
	int local_max_count=0, max_send_count=0;
	int sendNcopy[nprocs], rotate_index_array[nprocs], pos_status[nprocs];

	MPI_Type_size(sendtype, &typesize);

	w = ceil(log(nprocs) / float(log(r))); // calculate the number of digits when using r-representation
	nlpow = myPow(r, w-1); // maximum send number of elements
	d = (myPow(r, w) - nprocs) / nlpow; // calculate the number of highest digits

	if (rank == 0) { std::cout << "math: " << nprocs << " " << r << " " << w << " " << nlpow << " " << d << std::endl; }
//	double et = MPI_Wtime();
//	init_time = et - st;

//	st = MPI_Wtime();
	// 1. Find max send count
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	memcpy(sendNcopy, sendcounts, nprocs*sizeof(int));
//	et = MPI_Wtime();
//	findMax_time = et - st;


//	st = MPI_Wtime();
    // 2. create local index array after rotation
	for (int i = 0; i < nprocs; i++)
		rotate_index_array[i] = (2*rank-i+nprocs)%nprocs;
//	et = MPI_Wtime();
//	rotateIndex_time = et - st;

//	st = MPI_Wtime();
	// 3. exchange data with log(P) steps
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*nlpow);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*nlpow);
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

	int sent_blocks[nlpow];
	int di = 0, spoint = 1, distance = myPow(r, w-1), next_distance = distance*r;
//	et = MPI_Wtime();
//	alcCopy_time = et - st;

	for (int x = w-1; x > -1; x--) {
		int ze = (x == w - 1)? r - d: r;
		for (int z = ze-1; z > 0; z--) {

//			st = MPI_Wtime();
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
//			et = MPI_Wtime();
//			getBlock_time += et - st;

//			st = MPI_Wtime();
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
//			et = MPI_Wtime();
//			prepData_time += et - st;

//			st = MPI_Wtime();
			// 3) exchange metadata
			int recvrank = (rank + spoint) % nprocs; // receive data from rank - 2^step process
			int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process

			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0, metadata_recv, di, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);
//			et = MPI_Wtime();
//			excgMeta_time += et - st;

//			st = MPI_Wtime();
			for(int i = 0; i < di; i++)
				sendCount += metadata_recv[i];

			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recvrank, 1, comm, MPI_STATUS_IGNORE);
//			et = MPI_Wtime();
//			excgData_time += et - st;


//			st = MPI_Wtime();
			// 5) replaces
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];

				int origin_index = (sent_blocks[i] - rank + nprocs) % nprocs;
				if (origin_index % next_distance == (recvrank - rank + nprocs) % nprocs)
					memcpy(&recvbuf[rdispls[sent_blocks[i]]*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
				else
					memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				sendNcopy[send_index] = metadata_recv[i];
			}
//			et = MPI_Wtime();
//			replace_time += et - st;
		}
		distance /= r;
		next_distance /= r;
	}

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);

	return 0;
}

