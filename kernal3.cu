__global__ void kernelC3(double *A,double *B,double *C,int width, double r){
    int idy = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    
    int step;
    double prod_val = 0;
    if((idy>=(int)(width*(1-r)))||(idx>=(int)(width*r))) return;
    
    for(step=0;step<width;step++){
        prod_val += A[idy*width+step] * B[step*(int)(width*r)+idx];
    }
    
    
    C[idy*(int)(width*r)+idx] = prod_val;
}