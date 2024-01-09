/*
 * group_rbruck.cpp
 *
 *  Created on: Sep 1, 2022
 *      Author: kokofan
 */

#include "radix_r_bruck.h"

void uniform_isplit_r_bruck(int n, int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm) {

	int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

	if (nprocs % n > 0 || n >= nprocs) {
		if	(rank == 0)
			std::cout << "ERROR: the process count should be divided by the process count of a group." << std::endl;
		 MPI_Abort(comm, -1);
	}

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    int unit_size = sendcount * typesize;

	int ngroup = nprocs / n; // number of groups

	int sw = ceil(log(n) / log(r)); // required digits for intra-Bruck
	int gw = ceil(log(ngroup) / log(r)); // required digits for inter-Bruck
	int glpow = pow(r, gw-1); // the largest power of r that smaller than ngroup

	int grank = rank % n; // rank of each process in a group
	int gid = rank / n; // group id

	int max1 = glpow * n, max2 = pow(r, sw-1)*ngroup;
	int max_sd = (max1 > max2)? max1: max2; // max send data block count

	char* temp_buffer = (char*)malloc(max_sd * unit_size); // temporary buffer

	// Initial rotation phase for intra-Bruck
	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			int id = i * n + j;
			int rid = gsp + (2 * grank - j + n) % n;
			if (rid == id || recvbuf[id] == '1') { continue; }
			memcpy(temp_buffer, sendbuf+(id*unit_size), unit_size);
			memcpy(sendbuf+(id*unit_size), sendbuf+(rid*unit_size), unit_size);
			memcpy(sendbuf+(rid*unit_size), temp_buffer, unit_size);
			recvbuf[id] = '1';
			recvbuf[rid] = '1';
		}
	}

	int sent_blocks[max_sd];
	int di = 0, ci = 0;

	// Intra-Bruck
	int spoint = 1, distance = 1, next_distance = r;
	for (int x = 0; x < sw; x++) {
		for (int z = 1; z < r; z++) {
			di = 0; ci = 0;
			spoint = z * distance;
			if (spoint > n - 1) {break;}

			// get the sent data-blocks
			for (int g = 0; g < ngroup; g++) {
				for (int i = spoint; i < n; i += next_distance) {
					for (int j = i; j < (i+distance); j++) {
						if (j > n - 1 ) { break; }
						int id = g*n + (j + grank) % n;
						sent_blocks[di++] = id;
						memcpy(&temp_buffer[unit_size*ci++], &sendbuf[id*unit_size], unit_size);
					}
				}
			}

			// send and receive
			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			long long comm_size = di * unit_size;
			MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, recvbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			// replace with received data
			for (int i = 0; i < di; i++) {
				long long offset = sent_blocks[i] * unit_size;
				memcpy(sendbuf+offset, recvbuf+(i*unit_size), unit_size);
			}
		}
		distance *= r;
		next_distance *= r;
	}

    unit_size = n * sendcount * typesize;
	// Initial rotation phase for inter-Bruck
	for (int i = 0; i < ngroup; i++) {
		int index = (2 * gid - i + ngroup) % ngroup;
		memcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size);
	}

	// Inter-Bruck
	spoint = 1, distance = 1, next_distance = r;
    for (int x = 0; x < gw; x++) {
    	for (int z = 1; z < r; z++) {
    		spoint = z * distance;
			if (spoint > ngroup - 1) {break;}

			// get the sent data-blocks
			di = 0; ci = 0;
			for (int i = spoint; i < ngroup; i += next_distance) {
				for (int j = i; j < (i+distance); j++) {
					if (j > ngroup - 1 ) { break; }
					int id = (j + gid) % ngroup;
					sent_blocks[di++] = id;
					memcpy(&temp_buffer[unit_size*ci++], &recvbuf[id*unit_size], unit_size);
				}
			}

    		int recv_proc = (gid*n + (grank + spoint*n)) % nprocs; // receive data from rank - 2^step process
    		int send_proc = (gid*n + (grank - spoint*n + nprocs)) % nprocs; // send data from rank + 2^k process

    		long long comm_size = di * unit_size;
    		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, sendbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);

    		for (int i = 0; i < di; i++) {
    			long long offset = sent_blocks[i] * unit_size;
    			memcpy(recvbuf+offset, sendbuf+(i*unit_size), unit_size);
    		}
    	}
		distance *= r;
		next_distance *= r;
    }

	free(temp_buffer);
}


void uniform_inverse_isplit_r_bruck(int n, int r1, int r2, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm) {

	int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (r1 < 2 || r2 < 2) { r1 = 2; }

//    std::cout << nprocs << ", " << n << std::endl;

	if (nprocs % n > 0 || n >= nprocs) {
		if	(rank == 0)
			std::cout << "ERROR: the process count should be divided by the process count of a group." << std::endl;
		 MPI_Abort(comm, -1);
	}

    int typesize;
    MPI_Type_size(sendtype, &typesize);

    int unit_size = sendcount * typesize;

	int ngroup = nprocs / n; // number of groups

    if (r1 > n) { r1 = n; }
    if (r2 > ngroup) { r2 = ngroup; }

	int sw = ceil(log(n) / log(r1)); // required digits for intra-Bruck
	int sd = (pow(r1, sw) - n) / pow(r1, sw-1);

	int gw = ceil(log(ngroup) / log(r2)); // required digits for inter-Bruck
	int glpow = pow(r2, gw-1); // the largest power of r that smaller than ngroup
	int gd = (pow(r2, gw) - ngroup) / glpow;

	int grank = rank % n; // rank of each process in a group
	int gid = rank / n; // group id

	int max1 = glpow * n, max2 = pow(r1, sw-1)*ngroup;
	int max_sd = (max1 > max2)? max1: max2; // max send data block count

	char* temp_buffer = (char*)malloc(max_sd * unit_size); // temporary buffer

	// Initial rotation phase for intra-Bruck
	double st = MPI_Wtime();

	for (int i = 0; i < ngroup; i++) {
		int gsp = i*n;
		for (int j = 0; j < n; j++) {
			int id = i * n + j;
			int rid = gsp + (2 * grank - j + n) % n;
			if (rid == id || recvbuf[id] == '1') { continue; }
			memcpy(temp_buffer, sendbuf+(id*unit_size), unit_size);
			memcpy(sendbuf+(id*unit_size), sendbuf+(rid*unit_size), unit_size);
			memcpy(sendbuf+(rid*unit_size), temp_buffer, unit_size);
			recvbuf[id] = '1';
			recvbuf[rid] = '1';
		}
	}

	int sent_blocks[max_sd];
	int di = 0, ci = 0;

	// Intra-Bruck
	int spoint = 1, distance = myPow(r1, sw-1), next_distance = distance*r1;
	for (int x = sw-1; x > -1; x--) {
		int ze = r1 - 1;
		if (x == sw - 1) ze = r1 - 1 - sd;
		for (int z = ze; z > 0; z--) {
			di = 0; ci = 0;
			spoint = z * distance;

			// get the sent data-blocks
			for (int g = 0; g < ngroup; g++) {
				for (int i = spoint; i < n; i += next_distance) {
					for (int j = i; j < (i+distance); j++) {
						if (j > n - 1 ) { break; }
						int id = g*n + (j + grank) % n;
						sent_blocks[di++] = id;
						memcpy(&temp_buffer[unit_size*ci++], &sendbuf[id*unit_size], unit_size);
					}
				}
			}

			// send and receive
			int recv_proc = gid*n + (grank + spoint) % n; // receive data from rank + 2^step process
			int send_proc = gid*n + (grank - spoint + n) % n; // send data from rank - 2^k process

			long long comm_size = di * unit_size;
			MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, recvbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);

			// replace with received data
			for (int i = 0; i < di; i++) {
				long long offset = sent_blocks[i] * unit_size;
				memcpy(sendbuf+offset, recvbuf+(i*unit_size), unit_size);
			}
		}
		distance /= r1;
		next_distance /= r1;
	}

	double et = MPI_Wtime();
	intra_time = et - st;

	st = MPI_Wtime();
    unit_size = n * sendcount * typesize;
	// Initial rotation phase for inter-Bruck
	for (int i = 0; i < ngroup; i++) {
		int index = (2 * gid - i + ngroup) % ngroup;
		memcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size);
	}

	// Inter-Bruck
	spoint = 1, distance = myPow(r2, gw-1), next_distance = distance*r2;
    for (int x = gw-1; x > -1; x--) {
		int ze = r2 - 1;
		if (x == gw - 1) ze = r2 - 1 - gd;
    	for (int z = ze; z > 0; z--) {
    		spoint = z * distance;

			// get the sent data-blocks
			di = 0; ci = 0;
			for (int i = spoint; i < ngroup; i += next_distance) {
				for (int j = i; j < (i+distance); j++) {
					if (j > ngroup - 1 ) { break; }
					int id = (j + gid) % ngroup;
					sent_blocks[di++] = id;
					memcpy(&temp_buffer[unit_size*ci++], &recvbuf[id*unit_size], unit_size);
				}
			}

    		int recv_proc = (gid*n + (grank + spoint*n)) % nprocs; // receive data from rank - 2^step process
    		int send_proc = (gid*n + (grank - spoint*n + nprocs)) % nprocs; // send data from rank + 2^k process

    		long long comm_size = di * unit_size;
    		MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, sendbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);

    		for (int i = 0; i < di; i++) {
    			long long offset = sent_blocks[i] * unit_size;
    			memcpy(recvbuf+offset, sendbuf+(i*unit_size), unit_size);
    		}
    	}
		distance /= r2;
		next_distance /= r2;
    }
    et = MPI_Wtime();
    inter_time = et - st;

	free(temp_buffer);
}

