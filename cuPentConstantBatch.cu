// Andrew Gloster
// June 2019
// Solve batch pentadiagonal systems which all have the same matrix with variable entries
// Takes advantage of L2 cache hits

//   Copyright 2019 Andrew Gloster

//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at

//       http://www.apache.org/licenses/LICENSE-2.0

//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------

#include "cuPentConstantBatch.h"

// ---------------------------------------------------------------------
// Function to factorise the LHS matrix - Perform on CPU
// ---------------------------------------------------------------------

void pentFactorConstantBatch
(
	double* ds,  	// Array containing the lower diagonal, 2 away from the main diagonal. First two elements are 0.
	double* dl,  	// Array containing the lower diagonal, 1 away from the main diagonal. First elements is 0. 
	double* diag, 	 	// Array containing the main diagonal.
	double* du,	 	// Array containing the upper diagonal, 1 away from the main diagonal. Last element is 0.
	double* dw,  	// Array containing the upper diagonal, 2 awy from the main diagonal. Last 2 elements are 0.

	int nSolve  		// Size of the linear systems, number of unknowns
)
{
	// Indices used to store relative indexes
	int rowCurrent;
	int rowPrevious;
	int rowSecondPrevious;

	// Starting index
    rowCurrent = 0;

	// First Row
	diag[rowCurrent] = diag[rowCurrent];
	du[rowCurrent] = du[rowCurrent] / diag[rowCurrent];
	dw[rowCurrent] = dw[rowCurrent] / diag[rowCurrent];

	// Second row index
	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Second row
	dl[rowCurrent] = dl[rowCurrent];

	diag[rowCurrent] = diag[rowCurrent] - dl[rowCurrent] * du[rowPrevious];

	du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / diag[rowCurrent];

	dw[rowCurrent] = dw[rowCurrent] / diag[rowCurrent];

	// Interior rows - Note 0 indexing
	for (int i = 2; i < nSolve - 2; i++)
	{
		rowSecondPrevious = rowCurrent - 1; 
		rowPrevious = rowCurrent;
		rowCurrent += 1;

		dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
		
		diag[rowCurrent] = diag[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];

		dw[rowCurrent] = dw[rowCurrent] / diag[rowCurrent];

		du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / diag[rowCurrent];
	}

	// Second last row indexes
	rowSecondPrevious = rowCurrent - 1; 
	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Second last row
	dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
	diag[rowCurrent] = diag[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];
	du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / diag[rowCurrent];

	// Last row indexes
	rowSecondPrevious = rowCurrent - 1; 
	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Last row
	dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
	diag[rowCurrent] = diag[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];
}

// ---------------------------------------------------------------------
// Function to solve the Ax = b system of pentadiagonal matrices
// ---------------------------------------------------------------------

__global__ void pentSolveConstantBatch
(
	double* ds, 	// Array containing updated ds after using pentVariFactorBatch
	double* dl,		// Array containing updated ds after using pentVariFactorBatch
	double* diag,	// Array containing updated ds after using pentVariFactorBatch
	double* du,		// Array containing updated ds after using pentVariFactorBatchs
	double* dw,		// Array containing updated ds after using pentVariFactorBatch
	
	double* b,		// Dense array of RHS stored in interleaved format

	const int nSolve,  	// Size of the linear systems, number of unknowns
	const int nBatch	// Number of linear systems
)
{

	// Indices used to store relative indexes
	int rowCurrent;
	int rowPrevious;
	int rowSecondPrevious;

	int rowAhead;
	int rowSecondAhead;

	// Starting index
    rowCurrent = blockDim.x * blockIdx.x + threadIdx.x;

    // Constant matrix index
    int i = 0;

    // Only want to solve equations that exist
    if (rowCurrent < nBatch)
    {
    	// --------------------------
		// Forward Substitution
		// --------------------------

		// First Row
		b[rowCurrent] = b[rowCurrent] / diag[i];

		// Second row index
		rowPrevious = rowCurrent;
		rowCurrent += nBatch;
		i += 1;

		// Second row
		b[rowCurrent] = (b[rowCurrent] - dl[i] * b[rowPrevious]) / diag[i];

		// Interior rows - Note 0 indexing
		#pragma unroll
		for (i = 2; i < nSolve; i++)
		{
			rowSecondPrevious = rowCurrent - nBatch; 
			rowPrevious = rowCurrent;
			rowCurrent += nBatch;

			b[rowCurrent] = (b[rowCurrent] - ds[i] * b[rowSecondPrevious] - dl[i] * b[rowPrevious]) / diag[i];	
		}

    	// --------------------------
		// Backward Substitution
		// --------------------------

		// Last row
		b[rowCurrent] = b[rowCurrent];

		// Second last row index
		rowAhead = rowCurrent;
		rowCurrent -= nBatch;

		// Adjust for fact look has now become nSolve
		i = nSolve - 2;

		// Second last row
		b[rowCurrent] = b[rowCurrent] - du[i] * b[rowAhead];

		// Interior points - Note row indexing
		#pragma unroll
		for (i = nSolve - 3; i >= 0; i -= 1)
		{
			rowSecondAhead = rowCurrent + nBatch;
			rowAhead = rowCurrent;
			rowCurrent -= nBatch;

			b[rowCurrent] = b[rowCurrent] - du[i] * b[rowAhead] - dw[i] * b[rowSecondAhead];
		}
	}
}

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------