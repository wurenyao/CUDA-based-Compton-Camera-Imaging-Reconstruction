#include "Header.cuh"

#ifndef cpu_functions_header_h
#define cpu_functions_header_h

double GetLength(double z1, double y1, double x1, double z2, double y2, double x2);
void Print_Time_Complete( double clock_start , double clock_end , bool fin = 0 );

#endif