#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
//#include "../libs/mmio.h"
#include "../libs/mmio.c"


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

struct COO create_coo(struct COO coo, int* I, int* J, int M, int nz);
struct CSR coo_to_csr(struct COO coo);
void ElapsedTime(struct timeval t1, struct timeval t2);


int main(int argc, char* argv[]){

    /* mtx read */
    struct timeval start_time, end_time;
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

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


    /* reseve memory for matrices */
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));


    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d\n", &I[i], &J[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    //close file
    if (f !=stdin) fclose(f);
    

    //mtx file banner and sizes
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);

    struct COO coo = create_coo(coo, I, J, M, nz);
    struct CSR csr = coo_to_csr(coo);

    //csr format created, free space from coo    
    free(I);
    free(J);
    free(coo.cols);
    free(coo.rows);
    //gettimeofday(&start_time, NULL);
    //1-path 
    int* p1 = calloc(csr.nodes, sizeof(int));
    for(int i = 0; i < csr.nodes; i++){
        p1[i] = csr.pointers[i + 1] - csr.pointers[i];
    }
    
    //2-path
    int* p2 = calloc(csr.nodes, sizeof(int));
    int temp_index, start, end;
    for(int i = 0; i < csr.nodes; i++){
        start = csr.pointers[i];
        end = csr.pointers[i + 1];
        for(int j = start; j < end; j++){
            temp_index = csr.indices[j];
            p2[i] += p1[temp_index];
        }
        p2[i] = p2[i] - p1[i];
    }

    //bi-fork
    int* d3 = calloc(csr.nodes, sizeof(int));
    for(int i = 0; i < csr.nodes; i++){
        d3[i] = p1[i]*(p1[i] - 1) / 2;
    }

    //3-clique
    int temp_start, temp_end, temp_j, temp_k, count;
    int* c3 = calloc(csr.nodes, sizeof(int));
    for(int i = 0; i < csr.nodes; i++){
        count = 0;
        start = csr.pointers[i];
        end = csr.pointers[i + 1];
        for(int j = start; j < end; j++){
            temp_index = csr.indices[j];
            temp_start = csr.pointers[temp_index];
            temp_end = csr.pointers[temp_index + 1];
            temp_j = start;
            temp_k = temp_start;
            while(temp_j < end && temp_k < temp_end){
                if(csr.indices[temp_j] == csr.indices[temp_k]){
                    temp_j++;
                    temp_k++;
                    count++;
                }
                else if(csr.indices[temp_j] < csr.indices[temp_k]){
                    temp_j++;
                }
                else{
                    temp_k++;
                }
            }
        }
        c3[i] = (int) (count / 2);
    }

    //keep only the most complicated graphlet
    for(int i = 0; i < csr.nodes; i++){
        if(c3[i]){
            p2[i] -= 2*c3[i];
            d3[i] -= c3[i];
        }
    }
    // int kappa = csr.nodes - 1;
    // printf("%d %d %d %d %d\n", kappa, p1[kappa], p2[kappa], d3[kappa], c3[kappa]);


    free(p1);
    free(p2);
    free(d3);
    free(c3);
    free(csr.pointers);
    free(csr.indices);

    gettimeofday(&end_time, NULL);
    printf("Time: ");
    ElapsedTime(start_time, end_time);


    return 0;
}



void ElapsedTime(struct timeval t1, struct timeval t2){
    double time;
    time = t2.tv_sec - t1.tv_sec;
    time += (t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("%lf sec.\n", time);
}

struct CSR coo_to_csr(struct COO coo){

    struct CSR csr;
    csr.nodes = coo.nodes;
    csr.nz = coo.edges;
    csr.pointers = malloc((csr.nodes + 1) * sizeof(int));
    csr.indices = malloc(csr.nz * sizeof(int));
    
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
    coo.cols = calloc(coo.edges, sizeof(int));
    coo.rows = calloc(coo.edges, sizeof(int));
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


