#include "exo_matrix_RVV_fp32.h"

ukrFunction**** allocateMatrix() {
    ukrFunction**** matrix = (ukrFunction****)malloc(13 * sizeof(ukrFunction***));
    for (int i = 0; i < 13; i++) {
        matrix[i] = (ukrFunction***)malloc(7 * sizeof(ukrFunction**));
        for (int j = 0; j < 7; j++) {
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
    *matrix[0][3][0] = 	(ukrFunction)NULL;
    *matrix[0][3][1] = 	(ukrFunction)NULL;
    *matrix[0][4][0] = 	(ukrFunction)NULL;
    *matrix[0][4][1] = 	(ukrFunction)NULL;
    *matrix[0][5][0] = 	(ukrFunction)NULL;
    *matrix[0][5][1] = 	(ukrFunction)NULL;
    *matrix[0][6][0] = 	(ukrFunction)NULL;
    *matrix[0][6][1] = 	(ukrFunction)NULL;
    *matrix[1][0][0] = 	(ukrFunction)NULL;
    *matrix[1][0][1] = 	(ukrFunction)NULL;
    *matrix[1][1][0] = 	(ukrFunction)gemm_RVV_1x1_b0_col_fp32;
    *matrix[1][1][1] = 	(ukrFunction)gemm_RVV_1x1_b1_col_fp32;
    *matrix[1][2][0] = 	(ukrFunction)gemm_RVV_1x2_b0_col_fp32;
    *matrix[1][2][1] = 	(ukrFunction)gemm_RVV_1x2_b1_col_fp32;
    *matrix[1][3][0] = 	(ukrFunction)gemm_RVV_1x3_b0_col_fp32;
    *matrix[1][3][1] = 	(ukrFunction)gemm_RVV_1x3_b1_col_fp32;
    *matrix[1][4][0] = 	(ukrFunction)gemm_RVV_1x4_b0_col_fp32;
    *matrix[1][4][1] = 	(ukrFunction)gemm_RVV_1x4_b1_col_fp32;
    *matrix[1][5][0] = 	(ukrFunction)gemm_RVV_1x5_b0_col_fp32;
    *matrix[1][5][1] = 	(ukrFunction)gemm_RVV_1x5_b1_col_fp32;
    *matrix[1][6][0] = 	(ukrFunction)gemm_RVV_1x6_b0_col_fp32;
    *matrix[1][6][1] = 	(ukrFunction)gemm_RVV_1x6_b1_col_fp32;
    *matrix[2][0][0] = 	(ukrFunction)NULL;
    *matrix[2][0][1] = 	(ukrFunction)NULL;
    *matrix[2][1][0] = 	(ukrFunction)gemm_RVV_2x1_b0_col_fp32;
    *matrix[2][1][1] = 	(ukrFunction)gemm_RVV_2x1_b1_col_fp32;
    *matrix[2][2][0] = 	(ukrFunction)gemm_RVV_2x2_b0_col_fp32;
    *matrix[2][2][1] = 	(ukrFunction)gemm_RVV_2x2_b1_col_fp32;
    *matrix[2][3][0] = 	(ukrFunction)gemm_RVV_2x3_b0_col_fp32;
    *matrix[2][3][1] = 	(ukrFunction)gemm_RVV_2x3_b1_col_fp32;
    *matrix[2][4][0] = 	(ukrFunction)gemm_RVV_2x4_b0_col_fp32;
    *matrix[2][4][1] = 	(ukrFunction)gemm_RVV_2x4_b1_col_fp32;
    *matrix[2][5][0] = 	(ukrFunction)gemm_RVV_2x5_b0_col_fp32;
    *matrix[2][5][1] = 	(ukrFunction)gemm_RVV_2x5_b1_col_fp32;
    *matrix[2][6][0] = 	(ukrFunction)gemm_RVV_2x6_b0_col_fp32;
    *matrix[2][6][1] = 	(ukrFunction)gemm_RVV_2x6_b1_col_fp32;
    *matrix[3][0][0] = 	(ukrFunction)NULL;
    *matrix[3][0][1] = 	(ukrFunction)NULL;
    *matrix[3][1][0] = 	(ukrFunction)gemm_RVV_3x1_b0_col_fp32;
    *matrix[3][1][1] = 	(ukrFunction)gemm_RVV_3x1_b1_col_fp32;
    *matrix[3][2][0] = 	(ukrFunction)gemm_RVV_3x2_b0_col_fp32;
    *matrix[3][2][1] = 	(ukrFunction)gemm_RVV_3x2_b1_col_fp32;
    *matrix[3][3][0] = 	(ukrFunction)gemm_RVV_3x3_b0_col_fp32;
    *matrix[3][3][1] = 	(ukrFunction)gemm_RVV_3x3_b1_col_fp32;
    *matrix[3][4][0] = 	(ukrFunction)gemm_RVV_3x4_b0_col_fp32;
    *matrix[3][4][1] = 	(ukrFunction)gemm_RVV_3x4_b1_col_fp32;
    *matrix[3][5][0] = 	(ukrFunction)gemm_RVV_3x5_b0_col_fp32;
    *matrix[3][5][1] = 	(ukrFunction)gemm_RVV_3x5_b1_col_fp32;
    *matrix[3][6][0] = 	(ukrFunction)gemm_RVV_3x6_b0_col_fp32;
    *matrix[3][6][1] = 	(ukrFunction)gemm_RVV_3x6_b1_col_fp32;
    *matrix[4][0][0] = 	(ukrFunction)NULL;
    *matrix[4][0][1] = 	(ukrFunction)NULL;
    *matrix[4][1][0] = 	(ukrFunction)gemm_RVV_4x1_b0_col_fp32;
    *matrix[4][1][1] = 	(ukrFunction)gemm_RVV_4x1_b1_col_fp32;
    *matrix[4][2][0] = 	(ukrFunction)gemm_RVV_4x2_b0_col_fp32;
    *matrix[4][2][1] = 	(ukrFunction)gemm_RVV_4x2_b1_col_fp32;
    *matrix[4][3][0] = 	(ukrFunction)gemm_RVV_4x3_b0_col_fp32;
    *matrix[4][3][1] = 	(ukrFunction)gemm_RVV_4x3_b1_col_fp32;
    *matrix[4][4][0] = 	(ukrFunction)gemm_RVV_4x4_b0_col_fp32;
    *matrix[4][4][1] = 	(ukrFunction)gemm_RVV_4x4_b1_col_fp32;
    *matrix[4][5][0] = 	(ukrFunction)gemm_RVV_4x5_b0_col_fp32;
    *matrix[4][5][1] = 	(ukrFunction)gemm_RVV_4x5_b1_col_fp32;
    *matrix[4][6][0] = 	(ukrFunction)gemm_RVV_4x6_b0_col_fp32;
    *matrix[4][6][1] = 	(ukrFunction)gemm_RVV_4x6_b1_col_fp32;
    *matrix[5][0][0] = 	(ukrFunction)NULL;
    *matrix[5][0][1] = 	(ukrFunction)NULL;
    *matrix[5][1][0] = 	(ukrFunction)gemm_RVV_5x1_b0_col_fp32;
    *matrix[5][1][1] = 	(ukrFunction)gemm_RVV_5x1_b1_col_fp32;
    *matrix[5][2][0] = 	(ukrFunction)gemm_RVV_5x2_b0_col_fp32;
    *matrix[5][2][1] = 	(ukrFunction)gemm_RVV_5x2_b1_col_fp32;
    *matrix[5][3][0] = 	(ukrFunction)gemm_RVV_5x3_b0_col_fp32;
    *matrix[5][3][1] = 	(ukrFunction)gemm_RVV_5x3_b1_col_fp32;
    *matrix[5][4][0] = 	(ukrFunction)gemm_RVV_5x4_b0_col_fp32;
    *matrix[5][4][1] = 	(ukrFunction)gemm_RVV_5x4_b1_col_fp32;
    *matrix[5][5][0] = 	(ukrFunction)gemm_RVV_5x5_b0_col_fp32;
    *matrix[5][5][1] = 	(ukrFunction)gemm_RVV_5x5_b1_col_fp32;
    *matrix[5][6][0] = 	(ukrFunction)gemm_RVV_5x6_b0_col_fp32;
    *matrix[5][6][1] = 	(ukrFunction)gemm_RVV_5x6_b1_col_fp32;
    *matrix[6][0][0] = 	(ukrFunction)NULL;
    *matrix[6][0][1] = 	(ukrFunction)NULL;
    *matrix[6][1][0] = 	(ukrFunction)gemm_RVV_6x1_b0_col_fp32;
    *matrix[6][1][1] = 	(ukrFunction)gemm_RVV_6x1_b1_col_fp32;
    *matrix[6][2][0] = 	(ukrFunction)gemm_RVV_6x2_b0_col_fp32;
    *matrix[6][2][1] = 	(ukrFunction)gemm_RVV_6x2_b1_col_fp32;
    *matrix[6][3][0] = 	(ukrFunction)gemm_RVV_6x3_b0_col_fp32;
    *matrix[6][3][1] = 	(ukrFunction)gemm_RVV_6x3_b1_col_fp32;
    *matrix[6][4][0] = 	(ukrFunction)gemm_RVV_6x4_b0_col_fp32;
    *matrix[6][4][1] = 	(ukrFunction)gemm_RVV_6x4_b1_col_fp32;
    *matrix[6][5][0] = 	(ukrFunction)gemm_RVV_6x5_b0_col_fp32;
    *matrix[6][5][1] = 	(ukrFunction)gemm_RVV_6x5_b1_col_fp32;
    *matrix[6][6][0] = 	(ukrFunction)gemm_RVV_6x6_b0_col_fp32;
    *matrix[6][6][1] = 	(ukrFunction)gemm_RVV_6x6_b1_col_fp32;
    *matrix[7][0][0] = 	(ukrFunction)NULL;
    *matrix[7][0][1] = 	(ukrFunction)NULL;
    *matrix[7][1][0] = 	(ukrFunction)gemm_RVV_7x1_b0_col_fp32;
    *matrix[7][1][1] = 	(ukrFunction)gemm_RVV_7x1_b1_col_fp32;
    *matrix[7][2][0] = 	(ukrFunction)gemm_RVV_7x2_b0_col_fp32;
    *matrix[7][2][1] = 	(ukrFunction)gemm_RVV_7x2_b1_col_fp32;
    *matrix[7][3][0] = 	(ukrFunction)gemm_RVV_7x3_b0_col_fp32;
    *matrix[7][3][1] = 	(ukrFunction)gemm_RVV_7x3_b1_col_fp32;
    *matrix[7][4][0] = 	(ukrFunction)gemm_RVV_7x4_b0_col_fp32;
    *matrix[7][4][1] = 	(ukrFunction)gemm_RVV_7x4_b1_col_fp32;
    *matrix[7][5][0] = 	(ukrFunction)gemm_RVV_7x5_b0_col_fp32;
    *matrix[7][5][1] = 	(ukrFunction)gemm_RVV_7x5_b1_col_fp32;
    *matrix[7][6][0] = 	(ukrFunction)gemm_RVV_7x6_b0_col_fp32;
    *matrix[7][6][1] = 	(ukrFunction)gemm_RVV_7x6_b1_col_fp32;
    *matrix[8][0][0] = 	(ukrFunction)NULL;
    *matrix[8][0][1] = 	(ukrFunction)NULL;
    *matrix[8][1][0] = 	(ukrFunction)gemm_RVV_8x1_b0_col_fp32;
    *matrix[8][1][1] = 	(ukrFunction)gemm_RVV_8x1_b1_col_fp32;
    *matrix[8][2][0] = 	(ukrFunction)gemm_RVV_8x2_b0_col_fp32;
    *matrix[8][2][1] = 	(ukrFunction)gemm_RVV_8x2_b1_col_fp32;
    *matrix[8][3][0] = 	(ukrFunction)gemm_RVV_8x3_b0_col_fp32;
    *matrix[8][3][1] = 	(ukrFunction)gemm_RVV_8x3_b1_col_fp32;
    *matrix[8][4][0] = 	(ukrFunction)gemm_RVV_8x4_b0_col_fp32;
    *matrix[8][4][1] = 	(ukrFunction)gemm_RVV_8x4_b1_col_fp32;
    *matrix[8][5][0] = 	(ukrFunction)gemm_RVV_8x5_b0_col_fp32;
    *matrix[8][5][1] = 	(ukrFunction)gemm_RVV_8x5_b1_col_fp32;
    *matrix[8][6][0] = 	(ukrFunction)gemm_RVV_8x6_b0_col_fp32;
    *matrix[8][6][1] = 	(ukrFunction)gemm_RVV_8x6_b1_col_fp32;
    *matrix[9][0][0] = 	(ukrFunction)NULL;
    *matrix[9][0][1] = 	(ukrFunction)NULL;
    *matrix[9][1][0] = 	(ukrFunction)gemm_RVV_9x1_b0_col_fp32;
    *matrix[9][1][1] = 	(ukrFunction)gemm_RVV_9x1_b1_col_fp32;
    *matrix[9][2][0] = 	(ukrFunction)gemm_RVV_9x2_b0_col_fp32;
    *matrix[9][2][1] = 	(ukrFunction)gemm_RVV_9x2_b1_col_fp32;
    *matrix[9][3][0] = 	(ukrFunction)gemm_RVV_9x3_b0_col_fp32;
    *matrix[9][3][1] = 	(ukrFunction)gemm_RVV_9x3_b1_col_fp32;
    *matrix[9][4][0] = 	(ukrFunction)gemm_RVV_9x4_b0_col_fp32;
    *matrix[9][4][1] = 	(ukrFunction)gemm_RVV_9x4_b1_col_fp32;
    *matrix[9][5][0] = 	(ukrFunction)gemm_RVV_9x5_b0_col_fp32;
    *matrix[9][5][1] = 	(ukrFunction)gemm_RVV_9x5_b1_col_fp32;
    *matrix[9][6][0] = 	(ukrFunction)gemm_RVV_9x6_b0_col_fp32;
    *matrix[9][6][1] = 	(ukrFunction)gemm_RVV_9x6_b1_col_fp32;
    *matrix[10][0][0] = 	(ukrFunction)NULL;
    *matrix[10][0][1] = 	(ukrFunction)NULL;
    *matrix[10][1][0] = 	(ukrFunction)gemm_RVV_10x1_b0_col_fp32;
    *matrix[10][1][1] = 	(ukrFunction)gemm_RVV_10x1_b1_col_fp32;
    *matrix[10][2][0] = 	(ukrFunction)gemm_RVV_10x2_b0_col_fp32;
    *matrix[10][2][1] = 	(ukrFunction)gemm_RVV_10x2_b1_col_fp32;
    *matrix[10][3][0] = 	(ukrFunction)gemm_RVV_10x3_b0_col_fp32;
    *matrix[10][3][1] = 	(ukrFunction)gemm_RVV_10x3_b1_col_fp32;
    *matrix[10][4][0] = 	(ukrFunction)gemm_RVV_10x4_b0_col_fp32;
    *matrix[10][4][1] = 	(ukrFunction)gemm_RVV_10x4_b1_col_fp32;
    *matrix[10][5][0] = 	(ukrFunction)gemm_RVV_10x5_b0_col_fp32;
    *matrix[10][5][1] = 	(ukrFunction)gemm_RVV_10x5_b1_col_fp32;
    *matrix[10][6][0] = 	(ukrFunction)gemm_RVV_10x6_b0_col_fp32;
    *matrix[10][6][1] = 	(ukrFunction)gemm_RVV_10x6_b1_col_fp32;
    *matrix[11][0][0] = 	(ukrFunction)NULL;
    *matrix[11][0][1] = 	(ukrFunction)NULL;
    *matrix[11][1][0] = 	(ukrFunction)gemm_RVV_11x1_b0_col_fp32;
    *matrix[11][1][1] = 	(ukrFunction)gemm_RVV_11x1_b1_col_fp32;
    *matrix[11][2][0] = 	(ukrFunction)gemm_RVV_11x2_b0_col_fp32;
    *matrix[11][2][1] = 	(ukrFunction)gemm_RVV_11x2_b1_col_fp32;
    *matrix[11][3][0] = 	(ukrFunction)gemm_RVV_11x3_b0_col_fp32;
    *matrix[11][3][1] = 	(ukrFunction)gemm_RVV_11x3_b1_col_fp32;
    *matrix[11][4][0] = 	(ukrFunction)gemm_RVV_11x4_b0_col_fp32;
    *matrix[11][4][1] = 	(ukrFunction)gemm_RVV_11x4_b1_col_fp32;
    *matrix[11][5][0] = 	(ukrFunction)gemm_RVV_11x5_b0_col_fp32;
    *matrix[11][5][1] = 	(ukrFunction)gemm_RVV_11x5_b1_col_fp32;
    *matrix[11][6][0] = 	(ukrFunction)gemm_RVV_11x6_b0_col_fp32;
    *matrix[11][6][1] = 	(ukrFunction)gemm_RVV_11x6_b1_col_fp32;
    *matrix[12][0][0] = 	(ukrFunction)NULL;
    *matrix[12][0][1] = 	(ukrFunction)NULL;
    *matrix[12][1][0] = 	(ukrFunction)gemm_RVV_12x1_b0_col_fp32;
    *matrix[12][1][1] = 	(ukrFunction)gemm_RVV_12x1_b1_col_fp32;
    *matrix[12][2][0] = 	(ukrFunction)gemm_RVV_12x2_b0_col_fp32;
    *matrix[12][2][1] = 	(ukrFunction)gemm_RVV_12x2_b1_col_fp32;
    *matrix[12][3][0] = 	(ukrFunction)gemm_RVV_12x3_b0_col_fp32;
    *matrix[12][3][1] = 	(ukrFunction)gemm_RVV_12x3_b1_col_fp32;
    *matrix[12][4][0] = 	(ukrFunction)gemm_RVV_12x4_b0_col_fp32;
    *matrix[12][4][1] = 	(ukrFunction)gemm_RVV_12x4_b1_col_fp32;
    *matrix[12][5][0] = 	(ukrFunction)gemm_RVV_12x5_b0_col_fp32;
    *matrix[12][5][1] = 	(ukrFunction)gemm_RVV_12x5_b1_col_fp32;
    *matrix[12][6][0] = 	(ukrFunction)gemm_RVV_12x6_b0_col_fp32;
    *matrix[12][6][1] = 	(ukrFunction)gemm_RVV_12x6_b1_col_fp32;
}


void freeMatrix(ukrFunction**** matrix) {
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 7; j++) {
            for (int b = 0; b < 2; b++) {
                free(matrix[i][j][b]);
            }
            free(matrix[i][j]);
        }
        free(matrix[i]);
    }
    free(matrix);
}


