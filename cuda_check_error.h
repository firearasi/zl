#ifndef CUDA_CHECK_ERROR_H
#define CUDA_CHECK_ERROR_H

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

void __cudaSafeCall( cudaError err, const char *file, const int line );
void __cudaCheckError( const char *file, const int line );


#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


#endif // CUDA_CHECK_ERROR_H
