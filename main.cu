#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "kernel1.cu"
#include "kernel2.cu"
#include "kernel3.cu"
#include "kernel4.cu"




//Cuda runtime API function for Error handling
#define cudaErrorHandle() {                                                              
    cudaError_t e=cudaGetLastError();                                                   
    if(e!=cudaSuccess) {                                                               
        printf("Somethind is wrong in Cuda (Cuda failure) %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));   
        cudaDeviceReset();                                                              
        exit(EXIT_FAILURE);                                                             
    }                                                                                   
}


//Helper function to check the return values of malloc(), cudaManagedMalloc() and cudaMalloc()
#define Check_Allocation_Return_Value(a){   
    if(a==NULL) {                           
    printf(" There is a Cuda Allocation Error\n");           
    cudaDeviceReset();                      
    exit(EXIT_FAILURE);                     
    }                                       
}


// //general kernel(not used)
// __global__ void matrix_multiplication(double *A,double *B,double *C,int width){
//     int idy = blockIdx.y*blockDim.y+threadIdx.y;
//     int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
//     int step;
//     double prod_val = 0;
//     if((idy>=width)||((idx>=width))) return;
    
//     for(step=0;step<width;step++){
//         prod_val += A[idy*width+step] * B[step*width+idx];
//     }
    
//     C[idy*width+idx] = prod_val;
// }










int main(int argc,char *argv[]){
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    int N =7000;
    double *hA,*hB,*hC;
    int id,j,i;
    int ndev;
    double r = 0.5;
    double inv_r = (1-r);
    double *hA1,*hA2,*hB1,*hB2,*hC1,*hC2,*hC3,*hC4;
    double *dA1,*dA1_2,*dA2,*dA2_2,*dB1,*dB1_2,*dB2,*dB2_2;
    double *dC1,*dC2,*dC3,*dC4;
    
    printf("\nNumber of elements of the final matrix: %d\n",N * N);
    printf("Block 1 width: %d\n",(int)(N*r));
    printf("Block 2 width: %d\n",(int)(N*inv_r));
         
    cudaGetDeviceCount(&ndev);
    if(ndev==0){
        printf("NO GPU DEVICES AVAILABLE\n\n");
        exit(-1);
            
    }else{
        printf("Number of available GPUs: %d\n\n",ndev);
    }
        
    cudaMallocManaged(&hA,N*N*sizeof(double));
    Check_Allocation_Return_Value(hA)
    cudaMallocManaged(&hB,N*N*sizeof(double));
    Check_Allocation_Return_Value(hB)
    cudaMallocManaged(&hC,N*N*sizeof(double));
    Check_Allocation_Return_Value(hC)
    memset(hC,0,N*N*sizeof(double));
        
    srand (time(NULL));
    
    for(i=0;i<N*N;i++){
        hA[i] = rand()%10;
        hB[i] = rand()%10;
    }
    
    //Grid and block size initialization
    int grid_width = 1+N/32;
    dim3 dimGrid(grid_width,grid_width,1);
    dim3 dimBlock(32,32,1);
    
        
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    // kernel 1
    id=0;
    cudaSetDevice((int)(id%ndev));
    //cudaStreamCreate(&streams[id]);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    
    cudaMallocManaged(&hA1,(int)(N*N*r*sizeof(double)));
    Check_Allocation_Return_Value(hA1)
    cudaMallocManaged(&hB1,(int)(N*N*r*sizeof(double)));
    Check_Allocation_Return_Value(hB1)
    cudaMallocManaged(&hC1,(int)(N*N*r*r*sizeof(double)));
    Check_Allocation_Return_Value(hC1)
    
    for(int i=0;i<(int)(N*r);i++){
        for(int j=0;j<N;j++){
            hA1[i*N+j] =  hA[i*N+j];
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<(N*r);j++){
            hB1[i*(int)(N*r)+j] =  hB[i*N+j];
        }
    }

    cudaMalloc((void**)&dA1,(int)(N*N*r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dB1,(int)(N*N*r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dC1,(int)(N*N*r*r*sizeof(double)));
    cudaErrorHandle()
        
    // kernel 2
    id=1;
    cudaSetDevice((int)(id%ndev));
    //cudaStreamCreate(&streams[id]);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    
    cudaMallocManaged(&hB2,(int)(N*N*inv_r*sizeof(double)));
    Check_Allocation_Return_Value(hB2)
    cudaMallocManaged(&hC2,(int)(N*N*r*inv_r*sizeof(double)));
    Check_Allocation_Return_Value(hC2)
    
    for(int i=0;i<N;i++){
        for(int j=0;j<(N*inv_r);j++){
            hB2[i*(int)(N*inv_r)+j] =  hB[i*N+(int)(N*r)+j];
        }
    }
     
    cudaMalloc((void**)&dA1_2,(int)(N*N*r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dB2,(int)(N*N*inv_r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dC2,(int)(N*N*r*inv_r*sizeof(double)));
    cudaErrorHandle()
        
    // kernel 3
    id=2;
    cudaSetDevice(id%ndev);
    //cudaStreamCreate(&streams[id]);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    
    cudaMallocManaged(&hA2,(int)(N*N*inv_r*sizeof(double)));
    Check_Allocation_Return_Value(hA2)
    cudaMallocManaged(&hC3,(int)(N*N*inv_r*r*sizeof(double)));
    Check_Allocation_Return_Value(hC3)
    
    for(int i=0;i<(int)(N*inv_r);i++){
        for(int j=0;j<N;j++){
            hA2[i*N+j] =  hA[(i+(int)(N*r))*N+j];
        }
    }
    
    cudaMalloc((void**)&dA2,(int)(N*N*inv_r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dB1_2,(int)(N*N*r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dC3,(int)(N*N*r*inv_r*sizeof(double)));
    cudaErrorHandle()  
        
    // kernel 4
    id=3;
    cudaSetDevice(id%ndev);
    //cudaStreamCreate(&streams[id]);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);

    cudaMallocManaged(&hC4,(int)(N*N*inv_r*inv_r*sizeof(double)));
    Check_Allocation_Return_Value(hC4)
    
    cudaMalloc((void**)&dA2_2,(int)(N*N*inv_r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dB2_2,(int)(N*N*inv_r*sizeof(double)));
    cudaErrorHandle()
    cudaMalloc((void**)&dC4,(int)(N*N*inv_r*inv_r*sizeof(double)));
    cudaErrorHandle()
        
    //////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////
   
    printf("CPU-->GPU Memory copy(A1,B1,C1) - cudaMemcpyAsync\n");
    
    id=0;
    cudaSetDevice(id%ndev);
        
    cudaMemcpyAsync(dA1,hA1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()
    cudaMemcpyAsync(dB1,hB1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()
        
    printf(" Available Kernel 1 Execution...\n");
    availableKernel1 <<< dimGrid,dimBlock,0,streams[id]>>>(dA1,dB1,dC1,N,r);
    cudaErrorHandle()
    
    ///////////////////////////////////////////////////////////////////////////////  
    
    id=1;
    cudaSetDevice(id%ndev);
    
    printf("CPU-->GPU Memory copy(A1,B2,C2) - cudaMemcpyAsync\n");
    cudaMemcpyAsync(dA1_2,hA1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()
    cudaMemcpyAsync(dB2,hB2,(int)(N*N*inv_r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()
    
    printf("Available Kernel 2 Execution...\n");
    availableKernel2 <<< dimGrid,dimBlock,0,streams[id]>>>(dA1_2,dB2,dC2,N,r);
    cudaErrorHandle()
    
    ///////////////////////////////////////////////////////////////////////////////
    
    id=2;
    cudaSetDevice(id%ndev);
    
    printf("CPU-->GPU Memory copy(A2,B1,C3) - cudaMemcpyAsync\n");
    cudaMemcpyAsync(dA2,hA2,(int)(N*N*inv_r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()
    cudaMemcpyAsync(dB1_2,hB1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()
    
    printf("Available Kernel 3 Execution...\n");
    availableKernel3 <<< dimGrid,dimBlock,0,streams[id]>>>(dA2,dB1_2,dC3,N,r);
    cudaErrorHandle()

    ///////////////////////////////////////////////////////////////////////////////
    
    id=3;
    cudaSetDevice(id%ndev);
    
    printf("CPU-->GPU Memory copy(A2,B2,C4) - cudaMemcpyAsync\n");
    cudaMemcpyAsync(dA2_2,hA2,(int)(N*N*inv_r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()
    cudaMemcpyAsync(dB2_2,hB2,(int)(N*N*inv_r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    cudaErrorHandle()

    printf("Available Kernel 4 Execution...\n");
    availableKernel4 <<< dimGrid,dimBlock,0,streams[id]>>>(dA2_2,dB2_2,dC4,N,r);
    cudaErrorHandle()

    
    ///////////////////////////////////////////////////////////////////////////////
    printf("GPU-->CPU Memory copy (dC1) - cudaMemcpyAsync\n");
    cudaMemcpyAsync(hC1,dC1,(int)(N*N*r*r*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    cudaErrorHandle()
    
    printf("GPU-->CPU Memory copy (dC2) - cudaMemcpyAsync\n");
    cudaMemcpyAsync(hC2,dC2,(int)(N*N*r*inv_r*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    cudaErrorHandle()
    
    printf("GPU-->CPU Memory copy (dC3) - cudaMemcpyAsync\n");
    cudaMemcpyAsync(hC3,dC3,(int)(N*N*r*inv_r*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    cudaErrorHandle()
    
    printf("GPU-->CPU Memory copy (dC4) - cudaMemcpyAsync\n");
    cudaMemcpyAsync(hC4,dC4,(int)(N*N*inv_r*inv_r*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    cudaErrorHandle()
    
    
    //Synchronize in order to process the results of every invocation

    id=0;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);
    
    id=1;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);

    id=2;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);

    id=3;
    cudaSetDevice(id%ndev);
    cudaStreamSynchronize(streams[id]);

    //create the final Matrix
    for(i=0;i<(int)N*r;i++){
        for(j=0;j<(int)N*r;j++){
              hC[i*N+j] = hC1[i*(int)(N*r)+j];
              //printf("hC[%d]:%f ",i*N+j,hC[i*N+j]);
        }
        //printf("\n");
    }
    //printf("\n");
    
    
    for(i=0;i<(int)N*r;i++){
        for(j=0;j<(int)(N*inv_r);j++){
             hC[i*N+j+(int)(N*r)] = hC2[i*(int)(N*inv_r)+j];
             //printf("hC[%d]:%f",i*N+j+(int)(N*r),hC[i*N+j+(int)(N*r)]);
        }
        //printf("\n");
    }
    //printf("\n");
    
    for(i=0;i<(int)(N*inv_r);i++){
        for(j=0;j<(int)(N*r);j++){
             hC[(i+(int)(N*r))*N+j] = hC3[i*(int)(N*r)+j];
             //printf("hC[%d]:%f",(i+(int)(N*r))*N+j,hC[(i+(int)(N*r))*N+j]);
        }
        //printf("\n");
    }
    //printf("\n"); 
    
  
    for(i=0;i<(int)(N*inv_r);i++){
        for(j=0;j<(int)(N*inv_r);j++){
            hC[(i+(int)(N*r))*N+j+(int)(N*r)] = hC4[i*(int)(N*inv_r)+j];
          //  printf("hC[%d]:%f",(i+(int)(N*r))*N+j+(int)(N*r),hC[(i+(int)(N*r))*N+j+(int)(N*r)]);
        }
       // printf("\n");
    }
  //  printf("\n"); 
    
    
    
    // //Compare the GPU result with CPU computation(for validation)
    // printf("Check results...\n");
    // int k;
    // double res; 
    // for(i=0;i<N;i++){
    //     for(j=0;j<N;j++){
    //         res=0;
    //         for(k=0;k<N;k++){
    //             res+=hA[i*N+k]*hB[k*N+j];
    //         }
            
    //        //printf("%8.3f ",res);
    //        if(res != hC[i*N+j]){
    //             printf("NOT OK i:%d, j:%d\n",i,j);
    //             printf("true value:%f - computed value:%f\n\n",res,hC[i*N+j]);
    //        }
    //     }
    //     //printf("\n");
    // }
    
    
    
    
    printf("Free Unfied Memory and Device Memory\n");
    cudaFree(hA);
    cudaFree(hB);
    cudaFree(hC);
    cudaFree(hA1);
    cudaFree(hA2);
    cudaFree(hB1);
    cudaFree(hB2);
    cudaFree(hC1);
    cudaFree(hC2);
    cudaFree(hC3);
    cudaFree(hC4);
    
    id=0;
    cudaSetDevice(id%ndev);
    cudaFree(dA1);
    cudaErrorHandle()
    cudaFree(dB1);
    cudaErrorHandle()
    cudaFree(dC1);
    cudaErrorHandle()
    
    id=1;
    cudaSetDevice(id%ndev);
    cudaFree(dA1_2);
    cudaErrorHandle()
    cudaFree(dB2);
   cudaErrorHandle()
    cudaFree(dC2);
   cudaErrorHandle()
    
    id=2;
    cudaSetDevice(id%ndev);
    cudaFree(dA2);
    cudaErrorHandle()
    cudaFree(dB1_2);
   cudaErrorHandle()
    cudaFree(dC3);
    cudaErrorHandle()
    
    id=3;
    cudaSetDevice(id%ndev);
    cudaFree(dA2_2);
    cudaErrorHandle()
    cudaFree(dB2_2);
  cudaErrorHandle()
    cudaFree(dC4);
    cudaErrorHandle()
   
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cudaStreamDestroy(streams[3]);
    printf("Memory Free Done\n");
    return(0);
}
