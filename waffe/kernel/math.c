#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)
#define LPTA ((TSK*TSM)/(RTSM*RTSN))
#define LPTB ((TSK*TSN)/(RTSM*RTSN))


__kernel void matpluscols(const int M,
	const int N,
	const int K,
	__global floatX* A,
	__global floatX* b,
	__global floatX* C)
{
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N
	const int IDb = get_global_id(0) * K;

	if (ID0 < M && ID1 < N) {
		C[ID0 * N + ID1] = A[ID0 * N + ID1] + b[IDb];
	}
}



__kernel void matsum(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B,
	__global floatX* C)
{
	//    const int ID0 = get_global_id(0);
	//    const int ID1 = get_global_id(1);
	//    C[ID1 * M + ID0] = A[ID1 * M + ID0] + B[ID1 * M + ID0];
		 // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];
	__local floatX bufferB[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
		bufferB[tx][ty] = B[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		C[ID1 * M + ID0] = bufferA[tx][ty] + bufferB[tx][ty];
	}
}


__kernel void matk(const int M,
	const int N,
	const floatX k,
	__global floatX* A,
	__global floatX* B)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		B[ID1 * M + ID0] = bufferA[tx][ty] * k;
	}
}


__kernel void addk(const int M,
	const int N,
	const floatX k,
	__global floatX* A,
	__global floatX* B)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		B[ID1 * M + ID0] = bufferA[tx][ty] + k;
	}
}


__kernel void matsubstract(const int M,
	const int N,
	__global floatX* A,
	__global floatX* B,
	__global floatX* C)
{
	// // Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..M
	const int ID1 = get_group_id(1) * TS + ty; // 0..N

	// Set-up the local memory for shuffling
	__local floatX bufferA[TS][TS];
	__local floatX bufferB[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < M && ID1 < N) {
		bufferA[tx][ty] = A[ID1 * M + ID0];
		bufferB[tx][ty] = B[ID1 * M + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// Store the sum result (coalesced)
	if (ID0 < M && ID1 < N) {
		C[ID1 * M + ID0] = bufferA[tx][ty] - bufferB[tx][ty];
	}
}

__kernel void transpose(const int P, const int Q,
	__global floatX* input,
	__global floatX* output) {

	// Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0) * TS + tx; // 0..P
	const int ID1 = get_group_id(1) * TS + ty; // 0..Q

	// Set-up the local memory for shuffling
	__local floatX buffer[TS][TS];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < P && ID1 < Q) {
		buffer[ty][tx] = input[ID1 * P + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// We don't have to swap the x and y thread indices here,
	// because that's already done in the local memory
	const int newID0 = get_group_id(1) * TS + tx;
	const int newID1 = get_group_id(0) * TS + ty;

	// Store the transposed result (coalesced)
	if (newID0 < Q && newID1 < P) {
		output[newID1 * Q + newID0] = buffer[tx][ty];
	}
}