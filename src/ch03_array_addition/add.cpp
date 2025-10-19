#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double epsilon = 1.0e-15;
const double a = 1.05;
const double b = 2.65;
const double c = 5.34;

// the declaration of the functions
void add(const double *x, const double *y, double *z, const int n);
void check(const double *z, const int n);


int main(void){
    const int N = 1000;
    printf("Add two arrays of %d elements\n", N);

    const int M = sizeof(double) * N;  // the size of the array in bytes

    // allocate memory for the 3 arrays
    double *x = (double *)malloc(M);
    double *y = (double *)malloc(M);
    double *z = (double *)malloc(M);

    // initialize the arrays
    for (int i = 0; i < N; i++){
        x[i] = a;
        y[i] = b;
    }

    add(x, y, z, N);
    check(z, N);

    free(x);
    free(y);
    free(z);

    return 0;
}

void add(const double *x, const double *y, double *z, const int n){
    for (int i = 0; i < n; i++){
        z[i] = x[i] + y[i];
    }
}

void check(const double *z, const int n){
    for (int i = 0; i < n; i++){
        if (fabs(z[i] - c) > epsilon){  // we don't use `==`, we use `>`. when the difference is very small, we pass the check.
            printf("Error: z[%d] = %f\n", i, z[i]);
            exit(1);
        }
    }
}