#include "Header.cuh"

#ifndef gpu_functions_header_h
#define gpu_functions_header_h

__global__ void Simple_BP(double* cuda_p_density, double cos_com_sca_angle, double len_dis, double x_scattering, double y_scattering, double z_scattering, \
                          double x_absorbing, double y_absorbing, double z_absorbing, double xmin, double ymin, double zmin, double rec_x_voxel, double rec_y_voxel, \
                          double rec_z_voxel, double dtheta, int* event_count, int event_index);
__global__ void One_Batch_Integer_SysLis_Each_Event(int len_effective, int* cuda_system_list, long* cuda_integr_event_index, double* cuda_recon_density, \
                                                    double* cuda_amount_divisor);
__global__ void Multi_Batch_Integer_SysLis_Each_Event(const int len_effective, int* cuda_system_list, long* cuda_integr_event_index, double* cuda_recon_density, \
                                                      double* cuda_amount_divisor, int batch_index, const int amount_per_batch);

#endif