#include "exo_matrix_RVV_fp16.h"

ukrFunction**** allocateMatrix() {
    ukrFunction**** matrix = (ukrFunction****)malloc(9 * sizeof(ukrFunction***));
    for (int i = 0; i < 9; i++) {
        matrix[i] = (ukrFunction***)malloc(3 * sizeof(ukrFunction**));
        for (int j = 0; j < 3; j++) {
            matrix[i][j] = (ukrFunction**)malloc(2 * sizeof(ukrFunction*));
            for (int b = 0; b < 2; b++) {
                matrix[i][j][b] = (ukrFunction*)malloc(1 * sizeof(ukrFunction));
            }
        }
    }
    return matrix;
}


void fillMatrix(ukrFunction**** matrix) {
    *matrix[0][0][0] = 	(ukrFunction)NULL;
    *matrix[0][0][1] = 	(ukrFunction)NULL;
    *matrix[0][1][0] = 	(ukrFunction)NULL;
    *matrix[0][1][1] = 	(ukrFunction)NULL;
    *matrix[0][2][0] = 	(ukrFunction)NULL;
    *matrix[0][2][1] = 	(ukrFunction)NULL;
    *matrix[1][0][0] = 	(ukrFunction)NULL;
    *matrix[1][0][1] = 	(ukrFunction)NULL;
    *matrix[1][1][0] = 	(ukrFunction)gemm_RVV_1x1_b0_col_fp16;
    *matrix[1][1][1] = 	(ukrFunction)gemm_RVV_1x1_b1_col_fp16;
    *matrix[1][2][0] = 	(ukrFunction)gemm_RVV_1x2_b0_col_fp16;
    *matrix[1][2][1] = 	(ukrFunction)gemm_RVV_1x2_b1_col_fp16;
    *matrix[2][0][0] = 	(ukrFunction)NULL;
    *matrix[2][0][1] = 	(ukrFunction)NULL;
    *matrix[2][1][0] = 	(ukrFunction)gemm_RVV_2x1_b0_col_fp16;
    *matrix[2][1][1] = 	(ukrFunction)gemm_RVV_2x1_b1_col_fp16;
    *matrix[2][2][0] = 	(ukrFunction)gemm_RVV_2x2_b0_col_fp16;
    *matrix[2][2][1] = 	(ukrFunction)gemm_RVV_2x2_b1_col_fp16;
    *matrix[3][0][0] = 	(ukrFunction)NULL;
    *matrix[3][0][1] = 	(ukrFunction)NULL;
    *matrix[3][1][0] = 	(ukrFunction)gemm_RVV_3x1_b0_col_fp16;
    *matrix[3][1][1] = 	(ukrFunction)gemm_RVV_3x1_b1_col_fp16;
    *matrix[3][2][0] = 	(ukrFunction)gemm_RVV_3x2_b0_col_fp16;
    *matrix[3][2][1] = 	(ukrFunction)gemm_RVV_3x2_b1_col_fp16;
    *matrix[4][0][0] = 	(ukrFunction)NULL;
    *matrix[4][0][1] = 	(ukrFunction)NULL;
    *matrix[4][1][0] = 	(ukrFunction)gemm_RVV_4x1_b0_col_fp16;
    *matrix[4][1][1] = 	(ukrFunction)gemm_RVV_4x1_b1_col_fp16;
    *matrix[4][2][0] = 	(ukrFunction)gemm_RVV_4x2_b0_col_fp16;
    *matrix[4][2][1] = 	(ukrFunction)gemm_RVV_4x2_b1_col_fp16;
    *matrix[5][0][0] = 	(ukrFunction)NULL;
    *matrix[5][0][1] = 	(ukrFunction)NULL;
    *matrix[5][1][0] = 	(ukrFunction)gemm_RVV_5x1_b0_col_fp16;
    *matrix[5][1][1] = 	(ukrFunction)gemm_RVV_5x1_b1_col_fp16;
    *matrix[5][2][0] = 	(ukrFunction)gemm_RVV_5x2_b0_col_fp16;
    *matrix[5][2][1] = 	(ukrFunction)gemm_RVV_5x2_b1_col_fp16;
    *matrix[6][0][0] = 	(ukrFunction)NULL;
    *matrix[6][0][1] = 	(ukrFunction)NULL;
    *matrix[6][1][0] = 	(ukrFunction)gemm_RVV_6x1_b0_col_fp16;
    *matrix[6][1][1] = 	(ukrFunction)gemm_RVV_6x1_b1_col_fp16;
    *matrix[6][2][0] = 	(ukrFunction)gemm_RVV_6x2_b0_col_fp16;
    *matrix[6][2][1] = 	(ukrFunction)gemm_RVV_6x2_b1_col_fp16;
    *matrix[7][0][0] = 	(ukrFunction)NULL;
    *matrix[7][0][1] = 	(ukrFunction)NULL;
    *matrix[7][1][0] = 	(ukrFunction)gemm_RVV_7x1_b0_col_fp16;
    *matrix[7][1][1] = 	(ukrFunction)gemm_RVV_7x1_b1_col_fp16;
    *matrix[7][2][0] = 	(ukrFunction)gemm_RVV_7x2_b0_col_fp16;
    *matrix[7][2][1] = 	(ukrFunction)gemm_RVV_7x2_b1_col_fp16;
    *matrix[8][0][0] = 	(ukrFunction)NULL;
    *matrix[8][0][1] = 	(ukrFunction)NULL;
    *matrix[8][1][0] = 	(ukrFunction)gemm_RVV_8x1_b0_col_fp16;
    *matrix[8][1][1] = 	(ukrFunction)gemm_RVV_8x1_b1_col_fp16;
    *matrix[8][2][0] = 	(ukrFunction)gemm_RVV_8x2_b0_col_fp16;
    *matrix[8][2][1] = 	(ukrFunction)gemm_RVV_8x2_b1_col_fp16;
}


void freeMatrix(ukrFunction**** matrix) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 3; j++) {
            for (int b = 0; b < 2; b++) {
                free(matrix[i][j][b]);
            }
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}


