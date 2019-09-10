// Andrew Gloster
// June 2019
// Header file for user cuPentConstantBatch functions

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
// Define Header
// ---------------------------------------------------------------------

#ifndef PENTFUN_H
#define PENTFUN_H

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Programmer Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Header file functions
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Function to factorise the LHS matrix
// ---------------------------------------------------------------------

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
);

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
);

// ---------------------------------------------------------------------
// End of header file functions
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// End of definition
// ---------------------------------------------------------------------

#endif

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------
