#pragma once

#include <cstdint>

void launchGenerateSequence(int* d_output, int count);
void launchMultiplyIntArray(int* d_data, int count, int multiplier);

void launchSobelDerivativeXPass1(const uint8_t* d_input, float* d_temp, int width, int height, int channels);
void launchSobelDerivativeXPass2(const float* d_temp, float* d_output, int width, int height, int channels);
void launchSobelDerivativeYPass1(const uint8_t* d_input, float* d_temp, int width, int height, int channels);
void launchSobelDerivativeYPass2(const float* d_temp, float* d_output, int width, int height, int channels);
void launchSobelMagnitude(const uint8_t* d_input, const float* d_grad_x, const float* d_grad_y, uint8_t* d_output, int width, int height, int channels);

void launchGenerateRandomAndCount(float* d_random_values, int total_count, float threshold, int* d_counter);

void launchMatrixMultiply(const float* d_lhs, const float* d_rhs, float* d_output, int lhs_rows, int lhs_cols, int rhs_cols);
