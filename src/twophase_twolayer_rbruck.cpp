/*
 * group_rbruck.cpp
 *
 *  Created on: Sep 1, 2022
 *      Author: kokofan
 */

#include "rbruckv.h"

double init_time = 0, findMax_time = 0, rotateIndex_time = 0, alcCopy_time = 0,
getBlock_time = 0,prepData_time = 0, excgMeta_time = 0, excgData_time = 0, replace_time = 0,
orgData_time = 0, prepSP_time = 0, SP_time = 0;


int TTPL_rbruck_alltoallv(int n, int r, char *sendbuf, int *sendcounts,
									   int *sdispls, MPI_Datatype sendtype, char *recvbuf,
									   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double st = MPI_Wtime();
	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if (r > n) { r = n; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int ngroup, sw, gw, glpow;
	int grank, gid, max1, max2, max_sd;
	int local_max_count = 0, max_send_count = 0, id = 0;
	int updated_sentcouts[nprocs], rotate_index_array[nprocs], pos_status[nprocs];
	char *temp_send_buffer, *extra_buffer, *temp_recv_buffer;

	ngroup = ceil(nprocs / float(n)); // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / log(r)); // required digits for intra-Bruck


	gw = ceil(log(ngroup) / log(r)); // required digits for inter-Bruck
	glpow = pow(r, gw-1); // the largest power of r that smaller than ngroup

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	max2 = pow(r, sw-1) * ngroup;
	max1 = glpow * n;
	max_sd = (max1 > max2)? max1: max2; // max send data block count

	int sent_blocks[max_sd];
	double et = MPI_Wtime();
	init_time = et - st;

	st = MPI_Wtime();
	// 1. Find max send elements per data-block
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	et = MPI_Wtime();
	findMax_time = et - st;

	st = MPI_Wtime();
	// 2. create local index array after rotation
	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}
	et = MPI_Wtime();
	rotateIndex_time = et - st;

	st = MPI_Wtime();
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(updated_sentcouts, sendcounts, nprocs*sizeof(int));
	temp_send_buffer = (char*) malloc(max_send_count*typesize*max_sd);
	extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_sd);
	et = MPI_Wtime();
	alcCopy_time = et - st;

	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);

	// Intra-Bruck
	getBlock_time = 0, prepData_time = 0, excgMeta_time = 0, excgData_time = 0, replace_time = 0;
	int spoint = 1, distance = 1, next_distance = r, di = 0;
	for (int x = 0; x < sw; x++) {
		for (int z = 1; z < r; z++) {
			di = 0; spoint = z * distance;
			if (spoint > n - 1) {break;}

			st = MPI_Wtime();
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
			et = MPI_Wtime();
			getBlock_time += et - st;

			st = MPI_Wtime();
			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = updated_sentcouts[send_index];

				if (pos_status[send_index] == 0 )
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], updated_sentcouts[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], updated_sentcouts[send_index]*typesize);
				offset += updated_sentcouts[send_index]*typesize;
			}

			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			et = MPI_Wtime();
			prepData_time += et - st;

			st = MPI_Wtime();
			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }
			et = MPI_Wtime();
			excgMeta_time += et - st;

			st = MPI_Wtime();
			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);
			et = MPI_Wtime();
			excgData_time += et - st;

			st = MPI_Wtime();
			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				int origin_index = (sent_blocks[i] % n - grank + n) % n;
				if (sent_blocks[i] / n  == gid && origin_index < next_distance) {
					memcpy(&recvbuf[rdispls[sent_blocks[i]]*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
				}
				else {
					memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
				}

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				updated_sentcouts[send_index] = metadata_recv[i];
			}
			et = MPI_Wtime();
			replace_time += et - st;

		}
		distance *= r;
		next_distance *= r;
	}

	id = 0;
	int inter_rotate[ngroup];
	for (int i = 0; i < ngroup; i++) {
		inter_rotate[id++] = (2 * gid - i + ngroup) % ngroup;
	}

	// Inter-Bruck
	spoint = 1, distance = 1, next_distance = r;
    for (int x = 0; x < gw; x++) {
    	for (int z = 1; z < r; z++) {
    		spoint = z * distance;
    		if (spoint > ngroup - 1) {break;}

			// get the sent data-blocks
			di = 0;
			for (int i = spoint; i < ngroup; i += next_distance) {
				for (int j = i; j < (i+distance); j++) {
					if (j > ngroup - 1 ) { break; }
					int id = (j + gid) % ngroup;
					sent_blocks[di++] = id;
				}
			}

			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				metadata_send[i] = 0;
				id = inter_rotate[sent_blocks[i]];
				for (int j = 0; j < n; j++) {
					int ci = id*n + j;
					int dc = updated_sentcouts[rotate_index_array[ci]];
					if (ci % n == grank)
						memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[ci]*typesize], dc*typesize);
					else
						memcpy(&temp_send_buffer[offset], &extra_buffer[ci*max_send_count*typesize], dc*typesize);
					offset += dc*typesize;
					metadata_send[i] += dc;
				}
			}

			int recv_proc = (gid*n + (grank + spoint*n)) % nprocs; // receive data from rank - 2^step process
			int send_proc = (gid*n + (grank - spoint*n + nprocs)) % nprocs; // send data from rank + 2^k process

			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }

			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);

    	}
		distance *= r;
		next_distance *= r;
    }

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);

	return 0;
}


int TTPL_BT_alltoallv(int n, int r, char *sendbuf, int *sendcounts,
									   int *sdispls, MPI_Datatype sendtype, char *recvbuf,
									   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double st = MPI_Wtime();
	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if (r > n) { r = n; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int ngroup, sw;
	int grank, gid, imax, max_sd;
	int local_max_count = 0, max_send_count = 0, id = 0;
	int updated_sentcouts[nprocs], rotate_index_array[nprocs], pos_status[nprocs];
	char *temp_send_buffer, *extra_buffer, *temp_recv_buffer;

	ngroup = nprocs / float(n); // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / float(log(r))); // required digits for intra-Bruck

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	imax = myPow(r, sw-1) * ngroup;
	max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	if (rank == 0) {
		std::cout << "Math -- TTPL: " << nprocs << " " << r << " " << sw << " " <<  imax << " " << max_sd << std::endl;
	}

	int sent_blocks[max_sd];
	double et = MPI_Wtime();
	init_time = et - st;

	st = MPI_Wtime();
	// 1. Find max send elements per data-block
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	et = MPI_Wtime();
	findMax_time = et - st;

	st = MPI_Wtime();
	// 2. create local index array after rotation
	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}
	et = MPI_Wtime();
	rotateIndex_time = et - st;

	st = MPI_Wtime();
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(updated_sentcouts, sendcounts, nprocs*sizeof(int));
	temp_send_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_sd);
	et = MPI_Wtime();
	alcCopy_time = et - st;

	// Intra-Bruck
	getBlock_time = 0, prepData_time = 0, excgMeta_time = 0, excgData_time = 0, replace_time = 0;
	int spoint = 1, distance = 1, next_distance = r, di = 0;
	for (int x = 0; x < sw; x++) {
		for (int z = 1; z < r; z++) {
			di = 0; spoint = z * distance;
			if (spoint > n - 1) {break;}

			st = MPI_Wtime();
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
			et = MPI_Wtime();
			getBlock_time += et - st;

			st = MPI_Wtime();
			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = updated_sentcouts[send_index];

				if (pos_status[send_index] == 0 )
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], updated_sentcouts[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], updated_sentcouts[send_index]*typesize);
				offset += updated_sentcouts[send_index]*typesize;
			}

			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			et = MPI_Wtime();
			prepData_time += et - st;

			st = MPI_Wtime();
			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }
			et = MPI_Wtime();
			excgMeta_time += et - st;

			st = MPI_Wtime();
			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);
			et = MPI_Wtime();
			excgData_time += et - st;

			st = MPI_Wtime();
			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];

				memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				updated_sentcouts[send_index] = metadata_recv[i];
			}
			et = MPI_Wtime();
			replace_time += et - st;

		}
		distance *= r;
		next_distance *= r;
	}

	st = MPI_Wtime();
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
	et = MPI_Wtime();
	orgData_time = et - st;

	st = MPI_Wtime();
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
	et = MPI_Wtime();
	prepSP_time = et - st;

	st = MPI_Wtime();
	MPI_Request* req = (MPI_Request*)malloc(2*ngroup*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*ngroup*sizeof(MPI_Status));
	for (int i = 0; i < ngroup; i++) {

		int nsrc = (gid + i) % ngroup;
		int src =  nsrc * n + grank; // avoid always to reach first master node

		MPI_Irecv(&recvbuf[nrdisp[nsrc]], nrecv[nsrc]*typesize, MPI_CHAR, src, 0, comm, &req[i]);
	}

	for (int i = 0; i < ngroup; i++) {
		int ndst = (gid - i + ngroup) % ngroup;
		int dst = ndst * n + grank;

		MPI_Isend(&temp_send_buffer[nsdisp[ndst]], nsend[ndst]*typesize, MPI_CHAR, dst, 0, comm, &req[i+ngroup]);
	}

	MPI_Waitall(2*ngroup, req, stat);

	free(req);
	free(stat);

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);
	et = MPI_Wtime();
	SP_time = et - st;

	return 0;
}



int TTPL_BT_alltoallv_s1(int n, int r, char *sendbuf, int *sendcounts,
									   int *sdispls, MPI_Datatype sendtype, char *recvbuf,
									   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double st = MPI_Wtime();
	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if (r > n) { r = n; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int ngroup, sw;
	int grank, gid, imax, max_sd;
	int local_max_count = 0, max_send_count = 0, id = 0;
	int updated_sentcouts[nprocs], rotate_index_array[nprocs], pos_status[nprocs];
	char *temp_send_buffer, *extra_buffer, *temp_recv_buffer;

	ngroup = ceil(nprocs / float(n));  // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / float(log(r))); // required digits for intra-Bruck

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	imax = myPow(r, sw-1) * ngroup;
	max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	if (rank == 0) {
		std::cout << "Math -- TTPL-S1: " << nprocs << " " << r << " " << sw << " " <<  imax << " " << max_sd << std::endl;
	}

	int sent_blocks[max_sd];
	double et = MPI_Wtime();
	init_time = et - st;

	st = MPI_Wtime();
	// 1. Find max send elements per data-block
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	et = MPI_Wtime();
	findMax_time = et - st;

	st = MPI_Wtime();
	// 2. create local index array after rotation
	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}
	et = MPI_Wtime();
	rotateIndex_time = et - st;

	st = MPI_Wtime();
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(updated_sentcouts, sendcounts, nprocs*sizeof(int));
	temp_send_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_sd);
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);
	et = MPI_Wtime();
	alcCopy_time = et - st;

	// Intra-Bruck
	getBlock_time = 0, prepData_time = 0, excgMeta_time = 0, excgData_time = 0, replace_time = 0;
	int spoint = 1, distance = 1, next_distance = r, di = 0;
	for (int x = 0; x < sw; x++) {
		for (int z = 1; z < r; z++) {
			di = 0; spoint = z * distance;
			if (spoint > n - 1) {break;}

			st = MPI_Wtime();
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
			et = MPI_Wtime();
			getBlock_time += et - st;

			st = MPI_Wtime();
			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = updated_sentcouts[send_index];

				if (pos_status[send_index] == 0 )
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], updated_sentcouts[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], updated_sentcouts[send_index]*typesize);
				offset += updated_sentcouts[send_index]*typesize;
			}

			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			et = MPI_Wtime();
			prepData_time += et - st;

			st = MPI_Wtime();
			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }
			et = MPI_Wtime();
			excgMeta_time += et - st;

			st = MPI_Wtime();
			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);
			et = MPI_Wtime();
			excgData_time += et - st;

			st = MPI_Wtime();

			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				int origin_index = (sent_blocks[i] % n - grank + n) % n;
				if (sent_blocks[i] / n  == gid && origin_index < next_distance) {
					memcpy(&recvbuf[rdispls[sent_blocks[i]]*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
				}
				else {
					memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
				}

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				updated_sentcouts[send_index] = metadata_recv[i];
			}
			et = MPI_Wtime();
			replace_time += et - st;

		}
		distance *= r;
		next_distance *= r;
	}

	st = MPI_Wtime();
	int nquest = 2 * (ngroup-1) * n;
	MPI_Request* req = (MPI_Request*)malloc(nquest*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(nquest*sizeof(MPI_Status));

	nquest = 0;
	for (int i = 0; i < ngroup; i++) {
		int nsrc = (gid + i) % ngroup;
		if (nsrc == gid) { continue; }
		int src =  nsrc * n + grank;

		for (int j = 0; j < n; j++) {
			id = nsrc*n+j;
			MPI_Irecv(&recvbuf[rdispls[id]*typesize], recvcounts[id]*typesize, MPI_CHAR, src, j, comm, &req[nquest++]);
		}
	}

	for (int i = 0; i < ngroup; i++) {
		int ndst = (gid - i + ngroup) % ngroup;
		if (ndst == gid) { continue; }
		int dst = ndst * n + grank;

		for (int j = 0; j < n; j++) {
			id = ndst*n+j;
			int ds = updated_sentcouts[rotate_index_array[id]]*typesize;
			if (j == grank) {
				MPI_Isend(&sendbuf[sdispls[id]*typesize], ds, MPI_CHAR, dst, j, comm, &req[nquest++]);
			}
			else {
				MPI_Isend(&extra_buffer[id*max_send_count*typesize], ds, MPI_CHAR, dst, j, comm, &req[nquest++]);
			}
		}
	}
	MPI_Waitall(nquest, req, stat);

	free(req);
	free(stat);

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);
	et = MPI_Wtime();
	SP_time = et - st;

	return 0;
}


int TTPL_BT_alltoallv_s2(int n, int r, int bs, char *sendbuf, int *sendcounts,
									   int *sdispls, MPI_Datatype sendtype, char *recvbuf,
									   int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
	double st = MPI_Wtime();
	if ( r < 2 ) { return -1; }

	int rank, nprocs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	if (r > n) { r = n; }

	int typesize;
	MPI_Type_size(sendtype, &typesize);

	int ngroup, sw;
	int grank, gid, imax, max_sd;
	int local_max_count = 0, max_send_count = 0, id = 0;
	int updated_sentcouts[nprocs], rotate_index_array[nprocs], pos_status[nprocs];
	char *temp_send_buffer, *extra_buffer, *temp_recv_buffer;

	ngroup = ceil(nprocs / float(n));  // number of groups
    if (r > n) { r = n; }

	sw = ceil(log(n) / log(r)); // required digits for intra-Bruck

	grank = rank % n; // rank of each process in a group
	gid = rank / n; // group id
	imax = pow(r, sw-1) * ngroup;
	max_sd = (ngroup > imax)? ngroup: imax; // max send data block count

	int sent_blocks[max_sd];
	double et = MPI_Wtime();
	init_time = et - st;

	st = MPI_Wtime();
	// 1. Find max send elements per data-block
	for (int i = 0; i < nprocs; i++) {
		if (sendcounts[i] > local_max_count)
			local_max_count = sendcounts[i];
	}
	MPI_Allreduce(&local_max_count, &max_send_count, 1, MPI_INT, MPI_MAX, comm);
	et = MPI_Wtime();
	findMax_time = et - st;

	st = MPI_Wtime();
	// 2. create local index array after rotation
	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			rotate_index_array[id++] = gsp + (2 * grank - j + n) % n;
		}
	}
	et = MPI_Wtime();
	rotateIndex_time = et - st;

	st = MPI_Wtime();
	memset(pos_status, 0, nprocs*sizeof(int));
	memcpy(updated_sentcouts, sendcounts, nprocs*sizeof(int));
	temp_send_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	extra_buffer = (char*) malloc(max_send_count*typesize*nprocs);
	temp_recv_buffer = (char*) malloc(max_send_count*typesize*max_sd);
	memcpy(&recvbuf[rdispls[rank]*typesize], &sendbuf[sdispls[rank]*typesize], recvcounts[rank]*typesize);
	et = MPI_Wtime();
	alcCopy_time = et - st;

	// Intra-Bruck
	getBlock_time = 0, prepData_time = 0, excgMeta_time = 0, excgData_time = 0, replace_time = 0;
	int spoint = 1, distance = 1, next_distance = r, di = 0;
	for (int x = 0; x < sw; x++) {
		for (int z = 1; z < r; z++) {
			di = 0; spoint = z * distance;
			if (spoint > n - 1) {break;}

			st = MPI_Wtime();
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
			et = MPI_Wtime();
			getBlock_time += et - st;

			st = MPI_Wtime();
			// 2) prepare metadata and send buffer
			int metadata_send[di];
			int sendCount = 0, offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				metadata_send[i] = updated_sentcouts[send_index];

				if (pos_status[send_index] == 0 )
					memcpy(&temp_send_buffer[offset], &sendbuf[sdispls[send_index]*typesize], updated_sentcouts[send_index]*typesize);
				else
					memcpy(&temp_send_buffer[offset], &extra_buffer[sent_blocks[i]*max_send_count*typesize], updated_sentcouts[send_index]*typesize);
				offset += updated_sentcouts[send_index]*typesize;
			}

			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			et = MPI_Wtime();
			prepData_time += et - st;

			st = MPI_Wtime();
			// 3) exchange metadata
			int metadata_recv[di];
			MPI_Sendrecv(metadata_send, di, MPI_INT, send_proc, 0, metadata_recv, di, MPI_INT, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			for(int i = 0; i < di; i++) { sendCount += metadata_recv[i]; }
			et = MPI_Wtime();
			excgMeta_time += et - st;

			st = MPI_Wtime();
			// 4) exchange data
			MPI_Sendrecv(temp_send_buffer, offset, MPI_CHAR, send_proc, 1, temp_recv_buffer, sendCount*typesize, MPI_CHAR, recv_proc, 1, comm, MPI_STATUS_IGNORE);
			et = MPI_Wtime();
			excgData_time += et - st;

			st = MPI_Wtime();

			// 5) replace
			offset = 0;
			for (int i = 0; i < di; i++) {
				int send_index = rotate_index_array[sent_blocks[i]];
				int origin_index = (sent_blocks[i] % n - grank + n) % n;
				if (sent_blocks[i] / n  == gid && origin_index < next_distance) {
					memcpy(&recvbuf[rdispls[sent_blocks[i]]*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
				}
				else {
					memcpy(&extra_buffer[sent_blocks[i]*max_send_count*typesize], &temp_recv_buffer[offset], metadata_recv[i]*typesize);
				}

				offset += metadata_recv[i]*typesize;
				pos_status[send_index] = 1;
				updated_sentcouts[send_index] = metadata_recv[i];
			}
			et = MPI_Wtime();
			replace_time += et - st;

		}
		distance *= r;
		next_distance *= r;
	}


	st = MPI_Wtime();
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
	et = MPI_Wtime();
	orgData_time = et - st;

	st = MPI_Wtime();
	if (bs < 1 || bs > n) { bs = n; }
	int nblock = ceil( n / float(bs)), nmessage = ngroup*nblock;

//	if (rank == 0) {
//		std::cout << nblock << " " << nmessage << std::endl;
//	}

	int nsend[nmessage], nrecv[nmessage], nsdisp[nmessage], nrdisp[nmessage];
	int soffset = 0, roffset = 0;
	int tmessage = 0, bflag = 0;
	for (int i = 0; i < ngroup; i++) {
		for (int j = 0; j < nblock; j++) {
			id = i * nblock + j;
			nsend[id] = 0, nrecv[id] = 0;
			for (int k = 0; k < bs; k++) {
				int pid =  i * nblock*bs + j*bs + k;
				if (pid > nprocs - 1) { bflag = 1; break; }
				int sn = updated_sentcouts[rotate_index_array[pid]];
				nsend[id] += sn;
				nrecv[id] += recvcounts[id];
			}
			nsdisp[id] = soffset, nrdisp[id] = roffset;
			soffset += nsend[id] * typesize, roffset += nrecv[id] * typesize;
			tmessage++;
			if (bflag == 1) { break; }
		}
		if (bflag == 1) { break; }
	}
	et = MPI_Wtime();
	prepSP_time = et - st;

//
//	if (rank == 0) {
//		for (int i = 0; i < tmessage; i++) {
//			std::cout << nsend[i] << " " << nrecv[i] << " " << nsdisp[i] << " " << nrdisp[i] << std::endl;
//		}
//	}


//	int nquest = 2*(ngroup-1)*n;
//	MPI_Request* req = (MPI_Request*)malloc(nquest*sizeof(MPI_Request));
//	MPI_Status* stat = (MPI_Status*)malloc(nquest*sizeof(MPI_Status));
//
//	if (bs < 1 || bs > n) { bs = n; }
//
////	for (int i = 0; i < ngroup; i++) {
////		id = grank + i*n;
////		int ds = updated_sentcouts[rotate_index_array[id]]*typesize;
////		memcpy(&extra_buffer[id*max_send_count*typesize], &sendbuf[sdispls[id]*typesize], ds);
////	}
//
//

//
//
//
//

//
	st = MPI_Wtime();
	MPI_Request* req = (MPI_Request*)malloc(2*tmessage*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*tmessage*sizeof(MPI_Status));
	for (int i = 0; i < tmessage; i++) {

		int nsrc = (gid + i) % ngroup;
		int src =  nsrc * n + grank; // avoid always to reach first master node

		MPI_Irecv(&recvbuf[nrdisp[nsrc]], nrecv[nsrc]*typesize, MPI_CHAR, src, 0, comm, &req[i]);
	}

	for (int i = 0; i < tmessage; i++) {
		int ndst = (gid - i + ngroup) % ngroup;
		int dst = ndst * n + grank;

		MPI_Isend(&temp_send_buffer[nsdisp[ndst]], nsend[ndst]*typesize, MPI_CHAR, dst, 0, comm, &req[i+ngroup]);
	}

	MPI_Waitall(2*ngroup, req, stat);


//	nquest = 0;
//	for (int i = 0; i < ngroup; i++) {
//		int nsrc = (gid + i) % ngroup;
//		if (nsrc == gid) { continue; }
//		int src =  nsrc * n + grank;
//
//		for (int j = 0; j < nmessage; j++) {
//			int rcount = 0;
//			int rsp = nsrc*n + j;
//			for (int k = 0; k < bs; k++) {
//				id = nsrc*n + j*bs+k;
//				if (id > nprocs) { break; }
//				rcount += recvcounts[id];
//			}
//
//			MPI_Irecv(&recvbuf[rdispls[rsp]*typesize], rcount*typesize, MPI_CHAR, src, j, comm, &req[nquest++]);
//
////			if (rank == 0)
////			std::cout << rank << " " << i << " " << j << " " << rsp << " " << rcount << std::endl;
//		 }
//	}
//
//	for (int i = 0; i < ngroup; i++) {
//		int ndst = (gid - i + ngroup) % ngroup;
//		if (ndst == gid) { continue; }
//		int dst = ndst * n + grank;
//
//
//		for (int j = 0; j < nmessage; j++) {
//			int scount = 0;
//			int ssp = ndst*n + j;
//			for (int k = 0; k < bs; k++) {
//				id = ndst*n + j*bs+k;
//				if (id > nprocs) { break; }
//				scount += updated_sentcouts[rotate_index_array[id]]*typesize;
//			}
//
//			MPI_Isend(&extra_buffer[ssp*max_send_count*typesize], ds, MPI_CHAR, dst, j, comm, &req[nquest++]);
//
//			MPI_Irecv(&recvbuf[rdispls[rsp]*typesize], rcount*typesize, MPI_CHAR, src, j, comm, &req[nquest++]);
//
////			if (rank == 0)
////			std::cout << rank << " " << i << " " << j << " " << rsp << " " << rcount << std::endl;
//		 }
//
//		for (int j = 0; j < n; j++) {
//			id = ndst*n+j;
//			int ds = updated_sentcouts[rotate_index_array[id]]*typesize;
//			if (j == grank) {
//				MPI_Isend(&sendbuf[sdispls[id]*typesize], ds, MPI_CHAR, dst, j, comm, &req[nquest++]);
//			}
//			else {
//				MPI_Isend(&extra_buffer[id*max_send_count*typesize], ds, MPI_CHAR, dst, j, comm, &req[nquest++]);
//			}
//		}
//	}

//	MPI_Waitall(nquest, req, stat);

//	if (rank == 6) {
//		for (int i = 0; i < nprocs * max_send_count; i++) {
//			long long a;
//			memcpy(&a, &extra_buffer[i*typesize], typesize);
//			std::cout << a << std::endl;
//		}
//	}

	free(req);
	free(stat);

	free(temp_send_buffer);
	free(temp_recv_buffer);
	free(extra_buffer);
	et = MPI_Wtime();
	SP_time = et - st;

	return 0;
}


