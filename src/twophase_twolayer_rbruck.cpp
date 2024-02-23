/*
 * group_rbruck.cpp
 *
 *  Created on: Sep 1, 2022
 *      Author: kokofan
 */

#include "rbruckv.h"

int twophase_twolayer_rbruck_alltoallv(int n, int r, char *sendbuf, int *sendcounts,
									   int *sdispls, MPI_Datatype sendtype, char *recvbuf,
									   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{

	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	char *psnd, *prcv;

	if (r > n) { r = n; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int w = ceil(log(nprocs) / log(r)); // calculate the number of digits when using r-representation
	int nlpow = pow(r, w-1); // maximum send number of elements
	int d = (pow(r, w) - nprocs) / nlpow; // calculate the number of highest digits


	int ngroup = nprocs / n; // number of groups
    if (r > n) { r = n; }

	int sw = ceil(log(n) / log(r)); // required digits for intra-Bruck
	int sd = (pow(r, sw) - n) / pow(r, sw-1);

	int grank = rank % n; // rank of each process in a group
	int gid = rank / n; // group id
	int imax = pow(r, sw-1) * ngroup;
	int max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	// 1. Find max send elements per data-block
	int local_max_count = 0;
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	int max_send_count = 0;
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);

	int updated_sentcouts[nprocs];
	memcpy(updated_sentcouts, sendcounts, nprocs*sizeof(int));

	// 2. create local index array after rotation
	int rotate_index_array[nprocs];
	int id = 0;
	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}

	int pos_status[nprocs];
	memset(pos_status, 0, nprocs*sizeof(int));
	int sent_blocks[max_sd];
	int di = 0;

	char* temp_send_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	char* temp_recv_buffer = (char*) malloc(max_send_count*typesize*nlpow);

	// Intra-Bruck
	int spoint = 1, distance = 1, next_distance = r;
	for (int x = 0; x < sw; x++) {
		for (int z = 1; z < r; z++) {
			di = 0;
			spoint = z * distance;
			if (spoint > n - 1) {break;}

			// get the sent data-blocks
			for (int g = 0; g < ngroup; g++) {
				for (int i = spoint; i < n; i += next_distance) {
					for (int j = i; j < (i+distance); j++) {
						if (j > n - 1 ) { break; }
						int id = g*n + (j + grank) % n;
						sent_blocks[di++] = id;
					}
				}
			}

			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = updated_sentcouts[send_index];
				if (pos_status[send_index] == 0)
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], updated_sentcouts[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], updated_sentcouts[send_index]*typesize);
				offset += updated_sentcouts[send_index]*typesize;
			}

			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }

			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);

			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];

				memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				updated_sentcouts[send_index] = metadata_recv[i];
			}

		}
		distance *= r;
		next_distance *= r;
	}

	// organize data
	int index = 0;
	for (int i = 0; i < nprocs; i++) {
		int d = updated_sentcouts[rotate_index_array[i]]*typesize;
		if (grank == (i % n) ) {
			memcpy(&temp_send_buffer[index], &sendbuf[sdispls[i]*typesize], d);
		}
		else {
			memcpy(&temp_send_buffer[index], &extra_buffer[i*max_send_count*typesize], d);
		}
		index += d;
	}

	int nsend[ngroup], nrecv[ngroup], nsdisp[ngroup], nrdisp[ngroup];
	int soffset = 0, roffset = 0;
	for (int i = 0; i < ngroup; i++) {
		nsend[i] = 0, nrecv[i] = 0;
		for (int j = 0; j < n; j++) {
			int id = i * n + j;
			int sn = updated_sentcouts[rotate_index_array[id]];
			nsend[i] += sn;
			nrecv[i] += recvcounts[id];
		}
		nsdisp[i] = soffset, nrdisp[i] = roffset;
		soffset += nsend[i] * typesize, roffset += nrecv[i] * typesize;
	}

	MPI_Request* req = (MPI_Request*)malloc(2*ngroup*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*ngroup*sizeof(MPI_Status));
	for (int i = 0; i < ngroup; i++) {

		int nsrc = (gid + i) % ngroup;
		int src =  nsrc * n + grank; // avoid always to reach first master node

		prcv = (char *) recvbuf + nrdisp[nsrc];
		MPI_Irecv(prcv, nrecv[nsrc]*typesize, MPI_CHAR, src, 0, comm, &req[i]);
	}

	for (int i = 0; i < ngroup; i++) {
		int ndst = (gid - i + ngroup) % ngroup;
		int dst = ndst * n + grank;

		psnd = (char *) temp_send_buffer + nsdisp[ndst];
		MPI_Isend(psnd, nsend[ndst]*typesize, MPI_CHAR, dst, 0, comm, &req[i+ngroup]);
	}

	MPI_Waitall(2*ngroup, req, stat);

	free(req);
	free(stat);

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);

	return 0;
}


