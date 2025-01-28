/*
 * twophase_tunable_rbruckv.cpp
 *
 *  Created on: Jan 4, 2024
 *      Author: kokofan
 */

#include "rbruckv.h"

int tuna2_algorithm (int r, int b, char *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
		char *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm) {

	if ( r < 2 ) { r = 2; }

	int rank, nprocs, typesize;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);
	MPI_Type_size(sendtype, &typesize);

	if ( r > nprocs - 1 ) { r = nprocs - 1; }
	if (b <= 0 || b > nprocs) b = nprocs;

	int w, max_rank, nlpow, d, K, i, num_reqs;
	int local_max_count=0, max_send_count=0;
	int rotate_index_array[nprocs];
	w = 0, nlpow = 1, max_rank = nprocs - 1;

    while (max_rank) { w++; max_rank /= r; }   // number of bits required of r representation
    for (i = 0; i < w - 1; i++) { nlpow *= r; }   // maximum send number of elements
	d = (nlpow*r - nprocs) / nlpow; // calculate the number of highest digits
	K = w * (r - 1) - d; // the total number of communication rounds

	int rem1 = K + 1, rem2 = r + 1;
	int sendNcopy[nprocs - rem1];
	char *extra_buffer, *temp_recv_buffer;
	int extra_ids[nprocs - rem2];
	memset(extra_ids, -1, sizeof(extra_ids));
	int spoint = 1, distance = 1, next_distance = distance*r, di = 0;

	if (K < nprocs - 1) {
		// 1. Find max send count
		for (i = 0; i < nprocs; i++) {
			if (sendcounts[i] > local_max_count) { local_max_count = sendcounts[i]; }
		}
		MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

		// 2. create local index array after rotation
		for (i = 0; i < nprocs; i++) { rotate_index_array[i] = (2 * rank - i + nprocs) % nprocs; }

		// 3. exchange data with log(P) steps
		extra_buffer = (char*) malloc(max_send_count * typesize * (nprocs - rem1));
		temp_recv_buffer = (char*) malloc(max_send_count * nprocs * typesize);
	    if (extra_buffer == nullptr || temp_recv_buffer == nullptr) {
	        std::cerr << "extra_buffer or temp_recv_buffer allocation failed!" << std::endl;
	        return 1; // Exit program with error
	    }

		for (int x = 0; x < w; x++) {
			for (int z = 1; z < r; z ++) {
				spoint = z * distance;
				if (spoint > nprocs) { break; }
				int end = (spoint + distance > nprocs)? nprocs : spoint + distance;
				for (i = spoint + 1; i < end; i++) {
					extra_ids[i-rem2] = di++;
				}
			}
			distance *= r;
		}

	}

	// copy data that need to be sent to each rank itself
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

	int sent_blocks[r-1][nlpow];
	int metadata_recv[r-1][nlpow];
	int nc, rem, ns, ze, ss;
	spoint = 1, distance = 1, next_distance = distance*r;

	MPI_Request* reqs = (MPI_Request *) malloc(2 * b * sizeof(MPI_Request));
    if (reqs == nullptr) {
        std::cerr << "MPI_Requests allocation failed!" << std::endl;
        return 1; // Exit program with error
    }

	for (int x = 0; x < w; x++) {
		ze = (x == w - 1)? r - d: r;
		int zoffset = 0, zc = ze-1;
		int zns[zc];

		for (int k = 1; k < ze; k += b) {
			num_reqs = 0;
			ss = ze - k < b ? ze - k : b;
			for (i = 0; i < ss; i++) {

				int z = k + i;
				spoint = z * distance;
				nc = nprocs / next_distance * distance, rem = nprocs % next_distance - spoint;
				if (rem < 0) { rem = 0; }
				ns = (rem > distance)? (nc + distance) : (nc + rem);
				zns[z-1] = ns;

				int recvrank = (rank + spoint) % nprocs;
				int sendrank = (rank - spoint + nprocs) % nprocs; // send data from rank + 2^k process

				if (ns == 1) {
					MPI_Irecv(&recvbuf[rdispls[recvrank]*typesize], recvcounts[recvrank]*typesize, MPI_CHAR, recvrank, 1, comm, &reqs[num_reqs++]);

					MPI_Isend(&sendbuf[sdispls[sendrank]*typesize], sendcounts[sendrank]*typesize, MPI_CHAR, sendrank, 1, comm, &reqs[num_reqs++]);
				}
				else {
					di = 0;
					for (int i = spoint; i < nprocs; i += next_distance) {
						int j_end = (i+distance > nprocs)? nprocs: i+distance;
						for (int j = i; j < j_end; j++) {
							int id = (j + rank) % nprocs;
							sent_blocks[z-1][di++] = id;
						}
					}

					// 2) prepare metadata
					int metadata_send[di];
					int sendCount = 0, offset = 0;
					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];
						int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;

						if (i % distance == 0) {
							metadata_send[i] = sendcounts[send_index];
						}
						else {
							metadata_send[i] = sendNcopy[extra_ids[o]];
						}
						offset += metadata_send[i] * typesize;
					}

					MPI_Sendrecv(metadata_send, di, MPI_INT, sendrank, 0, metadata_recv[z-1], di, MPI_INT, recvrank, 0, comm, MPI_STATUS_IGNORE);

					for(int i = 0; i < di; i++) { sendCount += metadata_recv[z-1][i]; }

					// prepare send buffer
					char* temp_send_buffer = (char*) malloc(offset);

					// prepare send data
					offset = 0;
					for (int i = 0; i < di; i++) {
						int send_index = rotate_index_array[sent_blocks[z-1][i]];
						int o = (sent_blocks[z-1][i] - rank + nprocs) % nprocs - rem2;
						int size = 0;

						if (i % distance == 0) {
							size = sendcounts[send_index]*typesize;
							memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], size);
						}
						else {
							size = sendNcopy[extra_ids[o]]*typesize;
							memcpy(&temp_send_buffer[offset], &extra_buffer[extra_ids[o]*max_send_count*typesize], size);
						}
						offset += size;
					}


					MPI_Irecv(&temp_recv_buffer[zoffset], sendCount*typesize, MPI_CHAR, recvrank, 1, comm, &reqs[num_reqs++]);

					MPI_Isend(temp_send_buffer, offset, MPI_CHAR, sendrank, 1, comm, &reqs[num_reqs++]);

					zoffset += sendCount*typesize;

					free(temp_send_buffer);
				}
			}
			MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
		}

		// replaces
		int offset = 0;
		for (int i = 0; i < zc; i++) {
			for (int j = 0; j < zns[i]; j++){
				if (zns[i] > 1){
					int size = metadata_recv[i][j]*typesize;
					int o = (sent_blocks[i][j] - rank + nprocs) % nprocs - rem2;;

					if (j < distance) {
						memcpy(&recvbuf[rdispls[sent_blocks[i][j]]*typesize], &temp_recv_buffer[offset], size);
					}
					else {
						memcpy(&extra_buffer[extra_ids[o]*max_send_count*typesize], &temp_recv_buffer[offset], size);
						sendNcopy[extra_ids[o]] = metadata_recv[i][j];
					}
					offset += size;
				}
			}
		}

		distance *= r;
		next_distance *= r;
	}
	if (K < nprocs - 1) {
		free(extra_buffer);
		free(temp_recv_buffer);
	}
	free(reqs);

	return 0;
}

