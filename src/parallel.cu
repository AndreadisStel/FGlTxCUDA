#include <stdio.h>
#include <stdlib.h>
#include "../libs/mmio.c"
#include <sys/time.h>


//#define int unsigned int



struct COO{
    int* rows;
    int* cols;
    int nodes;
    int edges;
};

struct CSR{
    int* pointers;
    int* indices;
    int nodes;
    int nz;

};

__host__ struct COO create_coo(struct COO coo, int* I, int* J, int M, int nz);
__host__ struct CSR coo_to_csr(struct COO coo);

__host__ void ElapsedTime(timeval t1, timeval t2);

__global__ void Calc_p1(int* pointers, int* p1, int N){
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        p1[i] = pointers[i+1] - pointers[i]; 
    }
}

__global__ void Calc_d3(int* p1, int* d3, int* c3, int N){
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        d3[i] = p1[i]*(p1[i] - 1) / 2;
        if(c3[i])
            d3[i] -= c3[i];
    }
}

__global__ void Calc_p2(int* p1, int* p2, int* c3, int* pointers, int* indices, int N){
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        int start = pointers[i], end = pointers[i+1];
        for(int j = start; j < end; j++){
            p2[i] += p1[indices[j]];
        }
        p2[i] -= p1[i];
        if(c3[i])
            p2[i] -= 2*c3[i];
    }
}

__global__ void Calc_c3(int* c3, int* pointers, int* indices, int N){
    int i =  blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N){
        int start = pointers[i], end = pointers[i+1];
        int count = 0;
        int temp_j, temp_k, temp_index, temp_start, temp_end;
        for(int j = start; j < end; j++){
            temp_index = indices[j];
            temp_start = pointers[temp_index];
            temp_end = pointers[temp_index + 1];
            temp_j = start;
            temp_k = temp_start;
            while(temp_j < end && temp_k < temp_end){
                if(indices[temp_j] == indices[temp_k]){
                    temp_j++;
                    temp_k++;
                    count++;
                }
                else if(indices[temp_j] < indices[temp_k]){
                    temp_j++;
                }
                else{
                    temp_k++;
                }
            }
        }
        c3[i] = (int) (count / 2);
        
    }

}



int main(int argc, char* argv[]){

    // mtx read 
    timeval start_time, read_time, time, transform_time;  //measure elapsed time
    gettimeofday(&start_time, NULL);
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i, *I, *J;
    double *val;

    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    // find out size of sparse matrix

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    // reseve memory for matrices 
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));


    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d\n", &I[i], &J[i]);
        I[i]--;  // adjust from 1-based to 0-based 
        J[i]--;
    }

    //close file
    if (f !=stdin) fclose(f);
    
    
    //mtx file banner and sizes
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);

    gettimeofday(&read_time, NULL);
    printf("read time: ");
    ElapsedTime(start_time, read_time);


    struct COO coo = create_coo(coo, I, J, M, nz);
    struct CSR csr = coo_to_csr(coo);

    //csr format created, free space from coo    
    free(I);
    free(J);
    free(coo.cols);
    free(coo.rows);

    gettimeofday(&transform_time, NULL);
    printf("transform time: ");
    ElapsedTime(read_time, transform_time);



    int* d_pointers, *d_indices, *d_p1, *d_p2, *d_d3, *d_c3;    //device
    int *h_pointers, *h_indices, *h_p1, *h_p2, *h_d3, *h_c3;    //host
    int threadsPerBlock = 1024;
    int Blocks = (N + threadsPerBlock - 1)/threadsPerBlock;

    //allocate memory on host for results
    h_p1 = (int*) malloc(csr.nodes * sizeof(int));  
    h_p2 = (int*)malloc(csr.nodes * sizeof(int));
    h_d3 = (int*)malloc(csr.nodes * sizeof(int));
    h_c3 = (int*)malloc(csr.nodes * sizeof(int));
    h_pointers = csr.pointers;
    h_indices = csr.indices;
    
    
    cudaMalloc(&d_pointers, (csr.nodes + 1) * sizeof(int));
    cudaMemcpy(d_pointers, h_pointers, (csr.nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_p1, csr.nodes * sizeof(int));
    Calc_p1<<<Blocks, threadsPerBlock>>>(d_pointers, d_p1, csr.nodes);
    cudaMemcpy(h_p1, d_p1, csr.nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMalloc(&d_indices, csr.nz * sizeof(int));
    cudaMemcpy(d_indices, h_indices, csr.nz * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_c3, csr.nodes * sizeof(int));
    Calc_c3<<<Blocks, threadsPerBlock>>>(d_c3, d_pointers, d_indices, csr.nodes);
    cudaMemcpy(h_c3, d_c3, csr.nodes * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMalloc(&d_d3, csr.nodes * sizeof(int));
    Calc_d3<<<Blocks, threadsPerBlock>>>(d_p1, d_d3, d_c3, csr.nodes);
    cudaMemcpy(h_d3, d_d3, csr.nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMalloc(&d_p2, csr.nodes * sizeof(int));
    Calc_p2<<<Blocks, threadsPerBlock>>>(d_p1, d_p2, d_c3, d_pointers, d_indices, csr.nodes);
    cudaMemcpy(h_p2, d_p2, csr.nodes * sizeof(int), cudaMemcpyDeviceToHost);


    
    //free memory on device
    cudaFree(d_pointers);
    cudaFree(d_indices);
    cudaFree(d_c3);
    cudaFree(d_d3);
    cudaFree(d_p1);
    cudaFree(d_p2);
    
    //free memory on host
    free(csr.pointers);
    free(h_c3);
    free(h_d3);
    free(h_p1);
    free(h_p2);
    free(csr.indices);

    gettimeofday(&time, NULL);
    printf("exe time: ");
    ElapsedTime(transform_time, time);
    
    printf("Time: ");
    ElapsedTime(start_time, time);
    
    return 0;
}




void ElapsedTime(timeval t1, timeval t2){
    double time;
    time = t2.tv_sec - t1.tv_sec;
    time += (t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("%lf sec.\n", time);
}

struct CSR coo_to_csr(struct COO coo){

    struct CSR csr;
    csr.nodes = coo.nodes;
    csr.nz = coo.edges;
    csr.pointers = (int*)malloc((csr.nodes + 1) * sizeof(int));
    csr.indices = (int*)malloc(csr.nz * sizeof(int));
    
    for(int i = 0; i < csr.nodes; i++)
        csr.pointers[i] = 0;

    for(int i = 0; i < csr.nz; i++)
        csr.indices[i] = 0;

    for(int i = 0; i < csr.nz; i++)
        csr.pointers[coo.rows[i]]++;

    int temp = 0;
    int sum = 0;
    for(int i = 0; i < csr.nodes; i++){
        temp = csr.pointers[i];
        csr.pointers[i] = sum;
        sum += temp;
    }
    csr.pointers[csr.nodes] = csr.nz;

    int row = 0;
    int dest = 0;
    
    for(int i = 0; i < csr.nz; i++){
        row = coo.rows[i];
        dest = csr.pointers[row];
        csr.indices[dest] = coo.cols[i];
        csr.pointers[row]++;
    }
    
    temp = 0;
    int last = 0;
    
    for(int i = 0; i < csr.nodes + 1; i++){
        temp = csr.pointers[i];
        csr.pointers[i] = last;
        last = temp;
    }

    return csr;
}
struct COO create_coo(struct COO coo, int* I, int* J, int M, int nz){

    coo.nodes = M;
    coo.edges = 2 * nz;
    coo.cols = (int*)calloc(coo.edges, sizeof(int));
    coo.rows = (int*)calloc(coo.edges, sizeof(int));
    for(int i = 0; i < coo.edges; i++){
        if(i < nz){
            coo.cols[i] = J[i];
            coo.rows[i] = I[i];
        }
        else{
            coo.cols[i] = I[i - nz];
            coo.rows[i] = J[i - nz];
        }
    }
    
    return coo;
}


