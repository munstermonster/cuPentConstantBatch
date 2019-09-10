// Andrew Gloster
// June 2019
// Solve batch pentadiagonal systems which all have the same matrix
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

#include "cuPentUniformBatch.h"

// ---------------------------------------------------------------------
// Function to factorise the LHS matrix - Perform on CPU
// ---------------------------------------------------------------------

void pentFactorUniformBatch
(	
	double a,
	double b,
	double c,
	double d,
	double e,

	double* alpha,
	double* beta,
	double* gamma,
	double* delta,

	double nSolve
)
{
	// Set index to make life easier
	int i;

	// First position
	i = 0;

	alpha[i] = c;
	gamma[i] = d / alpha[i];
	delta[i] = e / alpha[i];

	// Second position
	i = 1;

	beta[i] = b;
	alpha[i] = c - beta[i] * gamma[i - 1];
	gamma[i] = (d - beta[i] * delta[i - 1]) / alpha[i];
	delta[i] = e / alpha[i];

	// Interior points
	for (int i = 2; i < nSolve - 2; i++)
	{
		beta[i] = b - a * gamma[i - 2];
		alpha[i] = c - a * delta[i - 2] - beta[i] * gamma[i - 1];
		gamma[i] = (d - beta[i] * delta[i - 1]) / alpha[i];
		delta[i] = e / alpha[i];
	}

	// Second last position
	i = nSolve - 2;

	beta[i] = b - a * gamma[i - 2];
	alpha[i] = c - a * delta[i - 2] - beta[i] * gamma[i - 1];
	gamma[i] = (d - beta[i] * delta[i - 1]) / alpha[i];
	
	// Last position
	i = nSolve - 1;

	beta[i] = b - a * gamma[i - 2];
	alpha[i] = c - a * delta[i - 2] - beta[i] * gamma[i - 1];
}

// ---------------------------------------------------------------------
// Function to solve batch of pentadiagonal matrices
// ---------------------------------------------------------------------

__global__ void pentSolveUniformBatch
(
	double a,

	double* alpha,
	double* beta,
	double* gamma,
	double* delta,
	
	double* bRHS,		

	int nSolve,  		
	int nBatch
)
{
	// Equation position index
	int eqIdxCurrent = blockDim.x * blockIdx.x + threadIdx.x;
	int eqIdxPrevious1;
	int eqIdxPrevious2;

    // Matrix position index
    int i;

 	if 	(eqIdxCurrent < nBatch)
	{
		// Forwards Sweep

	    // First position
	    i = 0;
		eqIdxCurrent = blockDim.x * blockIdx.x + threadIdx.x;
	    
	    bRHS[eqIdxCurrent] = bRHS[ eqIdxCurrent] / alpha[i];

	    // Second position
	    i = 1;
	    eqIdxPrevious1 = eqIdxCurrent;
	    eqIdxCurrent += nBatch;

	    bRHS[eqIdxCurrent] = (bRHS[eqIdxCurrent] - beta[i] * bRHS[eqIdxPrevious1]) / alpha[i];

	    // Loop
	    #pragma unroll
	    for (i = 2; i < nSolve; i++)
	    {
	    	// Move indexing
	    	eqIdxPrevious2 = eqIdxPrevious1;
	    	eqIdxPrevious1 = eqIdxCurrent;
	    	eqIdxCurrent += nBatch;

	    	// Forward substitute
	    	bRHS[eqIdxCurrent] = (bRHS[eqIdxCurrent] - a * bRHS[eqIdxPrevious2] - beta[i] * bRHS[eqIdxPrevious1]) / alpha[i];
	    }

	    // Backwards Sweep

	    // i is already where we want it
	    // The last position is solved

	    // Second last position
	    // Flip indexing
	    i = nSolve - 2;
	    eqIdxPrevious1 = eqIdxCurrent;
	    eqIdxCurrent -= nBatch;

	    bRHS[eqIdxCurrent] = bRHS[eqIdxCurrent] - gamma[i] * bRHS[eqIdxPrevious1];

	    // Loop 
	    for (i = nSolve - 3; i >= 0; i -= 1)
	    {
	    	// Move indexing
	    	eqIdxPrevious2 = eqIdxPrevious1;
	    	eqIdxPrevious1 = eqIdxCurrent;
	    	eqIdxCurrent -= nBatch;

	    	bRHS[eqIdxCurrent] = bRHS[eqIdxCurrent] - gamma[i] * bRHS[eqIdxPrevious1] - delta[i] * bRHS[eqIdxPrevious2];
	    }
	}
}

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------