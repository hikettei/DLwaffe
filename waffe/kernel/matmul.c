
// MAT OPERATORS
__kernel void mm_product(const int M,
	const int N,
	const int K,
	__global floatX* A,
	__global floatX* B,
	__global floatX* C)
{
	// Thread identifiers
	const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
	const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
	const int offsetM = TSM * get_group_id(0); // Work-group offset
	const int offsetN = TSN * get_group_id(1); // Work-group offset

	// Local memory to fit a tile of A and B
	__local floatX Asub[TSK][TSM];
	__local floatX Bsub[TSN][TSK + 2];

	// Allocate register space
	floatX Areg;
	floatX Breg[WPTN];
	floatX acc[WPTM][WPTN];

	// Initialise the accumulation registers
	for (int wm = 0; wm < WPTM; ++wm) {
		for (int wn = 0; wn < WPTN; ++wn) {
			acc[wm][wn] = 0.0f;
		}
	}

	// Loop over all tiles
	int numTiles = K / TSK;
	for (int t = 0; t < numTiles; ++t) {

		// Load one tile of A and B into local memory
		for (int la = 0; la < LPTA; ++la) {
			int tid = tidn * RTSM + tidm;
			int id = la * RTSN * RTSM + tid;
			int row = id % TSM;
			int col = id / TSM;
			int tiledIndex = TSK * t + col;
			int Aid = tiledIndex * M + offsetM + row;
			int Bid = tiledIndex * N + offsetN + row;
			Asub[col][row] = A[Aid];
			Bsub[row][col] = B[Bid];
		}

		// Synchronise to make sure the tile is loaded
		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop over the values of a single tile
		for (int k = 0; k < TSK; ++k) {

			// Cache the values of Bsub in registers
			for (int wn = 0; wn < WPTN; ++wn) {
				int col = tidn + wn * RTSN;
				Breg[wn] = Bsub[col][k];
			}

			// Perform the computation
			for (int wm = 0; wm < WPTM; ++wm) {
				int row = tidm + wm * RTSM;
				Areg = Asub[k][row];
				for (int wn = 0; wn < WPTN; ++wn) {
					acc[wm][wn] += Areg * Breg[wn];
				}
			}
		}

		// Synchronise before loading the next tile
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the final results in C
	for (int wm = 0; wm < WPTM; ++wm) {
		int globalRow = offsetM + tidm + wm * RTSM;
		for (int wn = 0; wn < WPTN; ++wn) {
			int globalCol = offsetN + tidn + wn * RTSN;
			C[globalCol * M + globalRow] = acc[wm][wn];
		}
	}
}

__kernel void matdot(const int M,
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
		C[ID1 * M + ID0] = bufferA[tx][ty] * bufferB[tx][ty];
	}
}

__kernel void mat_vec_product(const int M,
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
		C[ID1 * M + ID0] = bufferA[tx][ty] * bufferB[tx][ty];
	}
}