
__kernel void dsigmoid(const int M,
	const int N,
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
		B[ID1 * M + ID0] = bufferA[tx][ty] * (1 - bufferA[tx][ty]);
	}
}

