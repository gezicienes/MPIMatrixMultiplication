#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <chrono>



// Number of rowss and columnns in a matrix
#define N 5

MPI_Status status;

// Matrix holders are created
double matrix_a[N][N], matrix_b[N][N], matrix_c[N][N];

void serial()
{
    double a[N][N], b[N][N], mult[N][N];
   int i, j, k;

    srand(time(NULL));
    for (int mi = 0; mi < N; mi++) {
        for (int mj = 0; mj < N; mj++) {
            a[mi][mj] = rand() % 100;
            b[mi][mj] = rand() % 100;
        }
    }

    printf("\n\t\tMatrix - Matrix Multiplication using serial approach\n");

    printf("\nMatrix A\n\n");
    for (int mi = 0; mi < N; mi++) {
        for (int mj = 0; mj < N; mj++) {
            printf("%.0f\t", a[mi][mj]);
        }
        printf("\n");
    }

    printf("\nMatrix B\n\n");
    for (int mi = 0; mi < N; mi++) {
        for (int mj = 0; mj < N; mj++) {
            printf("%.0f\t", b[mi][mj]);
        }
        printf("\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Initializing elements of matrix mult to 0.
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
        {
            mult[i][j] = 0;
        }

    // Multiplying matrix a and b and storing in array mult.
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
            for (k = 0; k < N; ++k)
            {
                mult[i][j] += a[i][k] * b[k][j];
            }

    auto finish = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);

    // Displaying the multiplication of two matrix.
    printf("\nResult Matrix C = Matrix A * Matrix B:\n\n");
    for (int mi = 0; mi < N; mi++) {
        for (int mj = 0; mj < N; mj++) {
            printf("%.0f\t", mult[mi][mj]);
        }
        printf("\n");
    }
    printf("\n");

    printf("time elapsed: %lld microseconds", microseconds.count());

}

int main(int argc, char** argv)
{
    serial();


    int process_Count, processId, TaskCount, sources, desta, rowss, offset;

    // MPI environment is initialized
    MPI_Init(&argc, &argv);
    // Each process gets unique ID (rank)
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    // Number of processes in communicator will be assigned to variable -> process_Count
    MPI_Comm_size(MPI_COMM_WORLD, &process_Count);

    // Number of slave tasks will be assigned to variable -> TaskCount
    TaskCount = process_Count - 1;

    // Root (Master) process
    if (processId == 0) {

        // Matrix A and Matrix B both will be filled with random numbers
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix_a[i][j] = rand() % 100;
                matrix_b[i][j] = rand() % 100;
            }
        }

        printf("\n\t\tMatrix - Matrix Multiplication using MPI\n");

        // Print Matrix A
        printf("\nMatrix A\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f\t", matrix_a[i][j]);
            }
            printf("\n");
        }

        // Print Matrix B
        printf("\nMatrix B\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.0f\t", matrix_b[i][j]);
            }
            printf("\n");
        }

        
        rowss = N / TaskCount;
        
        offset = 0;

        
        auto start = std::chrono::high_resolution_clock::now();
        for (desta = 1; desta <= TaskCount; desta++)
        {
            
            MPI_Send(&offset, 1, MPI_INT, desta, 1, MPI_COMM_WORLD);
            
            MPI_Send(&rowss, 1, MPI_INT, desta, 1, MPI_COMM_WORLD);
            
            MPI_Send(&matrix_a[offset][0], rowss * N, MPI_DOUBLE, desta, 1, MPI_COMM_WORLD);
            
            MPI_Send(&matrix_b, N * N, MPI_DOUBLE, desta, 1, MPI_COMM_WORLD);

            
            offset = offset + rowss;
        }
        auto finish = std::chrono::high_resolution_clock::now();
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
        printf("time elapsed: %lld microseconds", microseconds.count());

       
        for (int i = 1; i <= TaskCount; i++)
        {
            sources = i;
            // Receive the offset of particular slave process
            MPI_Recv(&offset, 1, MPI_INT, sources, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rowss, 1, MPI_INT, sources, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&matrix_c[offset][0], rowss * N, MPI_DOUBLE, sources, 2, MPI_COMM_WORLD, &status);
        }

        // Print the result matrix
        printf("\nResult Matrix C = Matrix A * Matrix B:\n\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                printf("%.0f\t", matrix_c[i][j]);
            printf("\n");
        }
        printf("\n");
    }

    // Slave Processes
    if (processId > 0) {

        sources = 0;

 
        MPI_Recv(&offset, 1, MPI_INT, sources, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rowss, 1, MPI_INT, sources, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_a, rowss * N, MPI_DOUBLE, sources, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&matrix_b, N * N, MPI_DOUBLE, sources, 1, MPI_COMM_WORLD, &status);

        // Matrix multiplication

        for (int k = 0; k < N; k++) {
            for (int i = 0; i < rowss; i++) {
                matrix_c[i][k] = 0.0;
                for (int j = 0; j < N; j++)
                    matrix_c[i][k] = matrix_c[i][k] + matrix_a[i][j] * matrix_b[j][k];
            }
        }

        // value in matrix C
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);

        MPI_Send(&rowss, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&matrix_c, rowss * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}