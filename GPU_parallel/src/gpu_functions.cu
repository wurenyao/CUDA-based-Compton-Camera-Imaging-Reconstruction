#include "../include/Header.cuh"

// used functions
//****************************************************************************************************//
__global__ void Simple_BP(double* cuda_p_density, double cos_com_sca_angle, double len_dis, double x_scattering, double y_scattering, double z_scattering, \
                          double x_absorbing, double y_absorbing, double z_absorbing, double xmin, double ymin, double zmin, double rec_x_voxel, double rec_y_voxel, \
                          double rec_z_voxel, double dtheta, int* event_count, int event_index)
{
    unsigned index =  blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    
    double z_recon = zmin + blockIdx.y * rec_z_voxel + 0.5 * rec_z_voxel;
    double y_recon = ymin + blockIdx.x * rec_y_voxel + 0.5 * rec_y_voxel;
    double x_recon = xmin + threadIdx.x * rec_x_voxel + 0.5 * rec_x_voxel;
    // printf("position: %d, %d, %d \n", gridDim.y, gridDim.x, blockDim.x)
    // printf("position: %f, %f, %f \n",x_recon, y_recon, z_recon);
    // printf("index %d \n", index);
    // if (index >= len_effective) return;

    double len_recon = sqrt(pow((x_recon - x_scattering),2.0) + pow((y_recon - y_scattering),2.0) + pow((z_recon - z_scattering),2.0));
    double r_comptonAngle = ((x_recon - x_scattering) * (x_scattering - x_absorbing) + (y_recon - y_scattering) * (y_scattering - y_absorbing) + \
                            (z_recon - z_scattering) * (z_scattering - z_absorbing))/(len_recon*len_dis);
    double discT = abs(acos(r_comptonAngle) - acos(cos_com_sca_angle));
    if (discT < dtheta)
    {
        atomicAdd(&event_count[event_index], 1);
        cuda_p_density[index]++;
    }
}
//****************************************************************************************************//
__global__ void One_Batch_Integer_SysLis_Each_Event(int len_effective, int* cuda_system_list, long* cuda_integr_event_index, double* cuda_recon_density, \
                                                    double* cuda_amount_divisor)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len_effective) return;

    // shared memory can get more effciency than global memory
    extern __shared__ double s_f_tol_event[];
    s_f_tol_event[threadIdx.x] = 0.0;

    long bot_index = cuda_integr_event_index[index];
    long top_index = cuda_integr_event_index[index + 1];

    // calculate the density amount of pixels swept by each event(fitol)
    for (long i = bot_index; i < top_index; i++)
    {
        s_f_tol_event[threadIdx.x] += cuda_recon_density[cuda_system_list[i]];
    }

    // calculate the amount of 1 / fitol for each voxel
    for (long i = bot_index; i < top_index; i++)
    {
        atomicAdd(&(cuda_amount_divisor[cuda_system_list[i]]), (1.0 / s_f_tol_event[threadIdx.x]));  // atomicAdd for double only supported by compute capability 6.0 and higher 
    }
}
//****************************************************************************************************//
__global__ void Multi_Batch_Integer_SysLis_Each_Event(const int len_effective, int* cuda_system_list, long* cuda_integr_event_index, double* cuda_recon_density, \
                                                      double* cuda_amount_divisor, int batch_index, const int amount_per_batch)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned event_index = blockIdx.x * blockDim.x + threadIdx.x + batch_index * amount_per_batch;
    if (index >= amount_per_batch || event_index >= len_effective) return;

    // shared memory can get more effciency than global memory
    extern __shared__ double s_f_tol_event[];
    s_f_tol_event[threadIdx.x] = 0.0;

    long bot_index = cuda_integr_event_index[event_index] - cuda_integr_event_index[batch_index * amount_per_batch];
    long top_index = cuda_integr_event_index[event_index + 1] - cuda_integr_event_index[batch_index * amount_per_batch];

    // calculate the density amount of pixels swept by each event(fitol)
    for (long i = bot_index; i < top_index; i++)
    {
        s_f_tol_event[threadIdx.x] += cuda_recon_density[cuda_system_list[i]];
    }

    // calculate the amount of 1 / fitol for each voxel
    for (long i = bot_index; i < top_index; i++)
    {
        atomicAdd(&(cuda_amount_divisor[cuda_system_list[i]]), (1.0 / s_f_tol_event[threadIdx.x]));
    }
}
//****************************************************************************************************//