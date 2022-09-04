#include "../include/main_function_header.cuh"

// *** Input File *** //
#define Input_File "./True_CZT478.txt" 
//**********************************//

//const double Electron_Radius = 2.818;     // fm
const double Electron_Mass = 510.99;        // keV
const double Max_Ava_Video_Me = 10000;      // the maximum video memory that can be used (unit MiB)
//**********************************//
const double Gamma_Energy = 478.0;          // keV (the energy of incident photon)
const double Denergy = 3.0;                 // keV (the energy range for recorded events)
const double dtheta = 0.03;                 // rad ()
const int X_Bins = 100;                     // bins
const int Y_Bins = 100;                     // bins
const int Z_Bins = 100;                     // bins
const double xmin = -100.0;                 // mm
const double xmax = 100.0;                  // mm
const double ymin = -100.0;                 // mm
const double ymax = 100.0;                  // mm
const double zmin = -100.0;                 // mm
const double zmax = 100.0;                  // mm   
const double Distance_Filter = 10.0;        // mm  (0 means no Distance filter)
const int Compton_Ang_Filter = 0;           // mm  (0 means no Compton scattering angle filter)
//*********************************//
const int Events_Display = 1000;
const int MLEM_Itr = 40;
const int MLEM_Display = 4;

// Check whether CUDA-related functions are successfully executed
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

// define Compton event class
class Compton_event
{
public:
    Compton_event();
    Compton_event(double x1, double y1, double z1, double x2, double y2, double z2, double e1, double e2);
    ~Compton_event() {};
    bool Check(void);
    void GetComptonAngle(void);

    double x_scatter, y_scatter, z_scatter, e_scatter;
    double x_absorb, y_absorb, z_absorb, e_absorb;
    double cos_com_sca_angle;
    double com_sca_angle;
    double len_dis;
    int Check_ConicalVertex;
};
Compton_event::Compton_event()
{}
Compton_event::Compton_event(double x1, double y1, double z1, double x2, double y2, double z2, double e1, double e2)
{
    x_scatter = x1;
    y_scatter = y1;
    z_scatter = z1;
    e_scatter = e1;
    x_absorb = x2;
    y_absorb = y2;
    z_absorb = z2;
    e_absorb = e2;
    cos_com_sca_angle = 0;
    com_sca_angle = 0;
    len_dis = GetLength(z1, y1, x1, z2, y2, x2);
    Check_ConicalVertex = 0;
}
bool Compton_event::Check()
{
    double E_sca_max = (2 * Gamma_Energy * Gamma_Energy) / (Electron_Mass + 2 * Gamma_Energy);
    return (e_scatter < E_sca_max);
}
void Compton_event::GetComptonAngle()
{
    double cos_angle = 1 - (Electron_Mass*e_scatter)/(e_scatter*e_absorb + pow(e_absorb, 2));
    cos_com_sca_angle = cos_angle;
    // cout<<"康普顿角： "<<cos_com_sca_angle<<endl;

    if (Compton_Ang_Filter == 1)
    {
        if (cos_com_sca_angle == 1 || cos_com_sca_angle == -1 || cos_com_sca_angle < -0.5)  // angle filter
        {
            Check_ConicalVertex = 1;
        }
    }
}

int main()
{
    // Get the device id
    int device_id = 0;
    CHECK(cudaGetDevice(&device_id));

    // time start
    double time_start = clock();
    ifstream infile;
    infile.open(Input_File, ios::in);
    if (!infile) cout << "ifstream failure ! ! !" << endl;
    else cout << "ifstream good ! ! !" << endl;

    double rec_x_voxel = (xmax - xmin) / X_Bins;
    double rec_y_voxel = (ymax - ymin) / Y_Bins;
    double rec_z_voxel = (zmax - zmin) / Z_Bins;

    // ********** data importing ********** //
    double x1, x2, y1, y2, z1, z2, e1, e2;

    vector<Compton_event*> Compton_event_Set;
    while (!infile.eof())
    {
        infile >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> e1 >> e2;
        if (GetLength(z1, y1, x1, z2, y2, x2) < Distance_Filter) continue;  // distance filter
        if (abs((e1 + e2) - Gamma_Energy) > Denergy) continue;              // energy filter                                             

        Compton_event* event = new Compton_event(x1, y1, z1, x2, y2, z2, e1, e2);
        if (!(event->Check()))                                              // Compton edge filter
        {
            continue;
        }
        Compton_event_Set.push_back(event);
    }
    Compton_event_Set.pop_back();  // this is due to the last line of the input file is empty .
    // ********** End of importing file ********** //

    // print out data information
    const int len = Compton_event_Set.size();
    cout << "Compton Event number: " << len << endl;


    // ********** data preperation begining ********** //
    vector<double> cos_com_sca_angle_vec;      // get cosine value of Compton scattering angle
    vector<double> len_dis_vec;                // get distance value betewwn the two interaction position
    vector<double> x_scattering;
    vector<double> y_scattering;
    vector<double> z_scattering;
    vector<double> x_absorbing;
    vector<double> y_absorbing;
    vector<double> z_absorbing;               // position parameters

    int outside_num = 0;
    for (int index = 0; index < len; index++)
    {
        Compton_event_Set[index]->GetComptonAngle();
        if ((Compton_event_Set[index]->Check_ConicalVertex) == 1)
        {
            outside_num++;
            continue;
        }

        cos_com_sca_angle_vec.push_back((Compton_event_Set[index])->cos_com_sca_angle);
        len_dis_vec.push_back((Compton_event_Set[index])->len_dis);

        x_scattering.push_back((Compton_event_Set[index])->x_scatter);
        y_scattering.push_back((Compton_event_Set[index])->y_scatter);
        z_scattering.push_back((Compton_event_Set[index])->z_scatter);
        x_absorbing.push_back((Compton_event_Set[index])->x_absorb);
        y_absorbing.push_back((Compton_event_Set[index])->y_absorb);
        z_absorbing.push_back((Compton_event_Set[index])->z_absorb);
    }
    const int len_effective = len - outside_num;
    cout<<"Selected Compton event number: "<<len_effective<<endl;    // effective selected event, while not all of them are effective for reconstruction,
                                                                     // the useless events have no influence in the below iteration process

    // recnstruction space matrix
    double* cuda_p_density;      // single event recnstruction space matrix
    CHECK(cudaMalloc(&cuda_p_density, X_Bins*Y_Bins*Z_Bins*sizeof(double)));
    CHECK(cudaMemset(cuda_p_density, 0, X_Bins*Y_Bins*Z_Bins*sizeof(double)));

    double* cuda_recon_density;  // the recnstruction space matrix of all events
    CHECK(cudaMalloc(&cuda_recon_density, X_Bins*Y_Bins*Z_Bins*sizeof(double)));
    CHECK(cudaMemset(cuda_recon_density, 0, X_Bins*Y_Bins*Z_Bins*sizeof(double)));

    double* recon_density = new double[X_Bins*Y_Bins*Z_Bins]();   // the recnstruction space matrix of all events in CPU
    
    cout << "data pre-processing finished!!!" << endl;
    Print_Time_Complete(time_start, clock());
    cout << "**************************************************" << endl << endl;
    // ********** data preperation finished ********** //


    // ********** Begin of preprojection reconstruction **************//
    // ********** Simple backprojection reconstruction ********//
    // System matrix related
    int* cuda_event_count;
    int* event_count = new int[len_effective]();
    long* cuda_long_event_count;
    long* long_event_count = new long[len_effective]();
    memset(long_event_count, 0, len_effective * sizeof(long));
    CHECK(cudaMalloc(&cuda_event_count, len_effective * sizeof(int)));
    CHECK(cudaMemset(cuda_event_count, 0, len_effective * sizeof(int)));

    long* cuda_integr_event_index;
    long* integr_event_index = new long[len_effective + 1]();
    CHECK(cudaMalloc(&cuda_integr_event_index, (len_effective + 1) * sizeof(long)));
    CHECK(cudaMemset(cuda_integr_event_index, 0, (len_effective + 1) * sizeof(long)));

    vector<int> system_list; 
    // ****** //

    dim3 Grid_dim(Y_Bins,Z_Bins,1);
    for (int i = 0; i < len_effective; i++)
    {
        Simple_BP <<< Grid_dim, X_Bins >>> (cuda_p_density, cos_com_sca_angle_vec[i], len_dis_vec[i], x_scattering[i], y_scattering[i], z_scattering[i], \
                                            x_absorbing[i], y_absorbing[i], z_absorbing[i], xmin, ymin, zmin, rec_x_voxel, rec_y_voxel, rec_z_voxel, dtheta, \
                                            cuda_event_count, i);
        cudaDeviceSynchronize();

        CHECK(cudaMemcpy(recon_density, cuda_p_density, X_Bins*Y_Bins*Z_Bins*sizeof(double), cudaMemcpyDeviceToHost));
        for (int k = 0; k < Z_Bins; k++)
        {
            for (int j = 0; j < Y_Bins; j++)
            {
                for (int i = 0; i < X_Bins; i++)
                {
                    if (recon_density[k*X_Bins*Y_Bins + j*X_Bins + i] == 1) system_list.push_back(k*X_Bins*Y_Bins + j*X_Bins + i);
                }
            }
        }
        thrust::transform(thrust::device, cuda_p_density, cuda_p_density + X_Bins * Y_Bins * Z_Bins, cuda_recon_density, cuda_recon_density, thrust::plus<double>());
        CHECK(cudaMemset(cuda_p_density, 0, X_Bins * Y_Bins * Z_Bins * sizeof(double)));

        if (i == (len_effective - 1)) cout << len_effective << " events end  !!!" << endl;
        if (i % Events_Display == 0) cout<<i<<" events end  !!!" << endl;
    }
    // System matrix related
    CHECK(cudaMemcpy(event_count, cuda_event_count, len_effective * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < len_effective; i++) long_event_count[i] = event_count[i];
    CHECK(cudaMalloc(&cuda_long_event_count, len_effective * sizeof(long)));
    CHECK(cudaMemcpy(cuda_long_event_count, long_event_count, len_effective * sizeof(long), cudaMemcpyHostToDevice));

    thrust::inclusive_scan(thrust::device, cuda_long_event_count, cuda_long_event_count + len_effective, cuda_integr_event_index + 1);
    CHECK(cudaMemcpy(integr_event_index, cuda_integr_event_index, (len_effective + 1)*sizeof(long), cudaMemcpyDeviceToHost));
    cout<<"Total system parameter: "<<integr_event_index[len_effective]<<endl;
    // ****** //

    // deal with video memory
    const long long Max_Ava_int_num = (Max_Ava_Video_Me * 1024 * 1024) / 4;
    const int parameter_per_event = (integr_event_index[len_effective] / len_effective) + 1;   // the average parameter number (intersection voxel number) for the total Compton event
    const int amount_per_batch = (Max_Ava_int_num / parameter_per_event) - 1;                  // the number of Compton event that can be deal with in one batch
    const int iteration_batch_num = (len_effective / amount_per_batch) + 1;                    // the batch number cooresponding to the iteration process
    cout<<"MLEM iteration batch number: "<<iteration_batch_num<<endl;
    if (iteration_batch_num != 1) cout<<"The maximum event for one batch iteration is: "<<amount_per_batch<<endl;
    // ****** //

    cout << "Simple backprojection reconstruction finished!!!" << endl;
    // ********** End of simple backprojection reconstruction ********** //

    // Check cuda kernal function
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    if (MLEM_Itr != 0) cout << "System matrix calculation finished!!!" << endl;
    else cout << "No need of system matrix!!!" << endl;
    Print_Time_Complete(time_start, clock());
    cout << "**************************************************" << endl << endl;
    // ********** End of preprojection reconstruction ********** //
    

    // ********** Destroy the useless variables ******************** //
    CHECK(cudaFree(cuda_p_density));
    CHECK(cudaFree(cuda_event_count));
    CHECK(cudaFree(cuda_long_event_count));
    // ********** End of destroying the useless variables ********** //


    // ********** Simple Back Projection Output ********** //
    // write out into file
    CHECK(cudaMemcpy(recon_density, cuda_recon_density, X_Bins*Y_Bins*Z_Bins*sizeof(double), cudaMemcpyDeviceToHost));
    std::ofstream outList;
	outList.open("Simple_back_projection.txt",ios::out);
    if(!outList) cout<<"Open the file failure..\n"<<endl;

    for (int k = 0; k < Z_Bins; k++)
    {
        for (int j = 0; j < Y_Bins; j++)
        {
            for (int i = 0; i < X_Bins; i++)
            {
                outList<<recon_density[k*X_Bins*Y_Bins + j*X_Bins + i]<<" ";
            }
            outList<<endl;
        }
        outList<<endl;
    }
    outList.close();
    cout<<"Simple back projection Output File if OK ! ! !"<<endl;
    Print_Time_Complete( time_start , clock() );
    cout<<"**************************************************"<<endl << endl;
    // ********** End of Back Projection ********** //


    // ********************* MLEM itreation ********************** //
    int itr_num = 0;
    double TDetEvent;

    double* cuda_amount_divisor;
    CHECK(cudaMalloc(&cuda_amount_divisor, X_Bins * Y_Bins * Z_Bins * sizeof(double)));
    CHECK(cudaMemset(cuda_amount_divisor, 0, X_Bins * Y_Bins * Z_Bins * sizeof(double)));

    if (iteration_batch_num == 1) // One MLEM iteration batch mode
    {
        int* cuda_system_list;
        CHECK(cudaMalloc(&cuda_system_list, (integr_event_index[len_effective]) * sizeof(int)));
        CHECK(cudaMemcpy(cuda_system_list, &system_list[0], (integr_event_index[len_effective]) * sizeof(int), cudaMemcpyHostToDevice));

        const int MLEM_block_dim = 128;
        int MLEM_grid_dim;
        if (len_effective % MLEM_block_dim == 0) MLEM_grid_dim = int(len_effective / MLEM_block_dim);
        else MLEM_grid_dim = int(len_effective / MLEM_block_dim) + 1;

        for (int k = 0; k < MLEM_Itr; k++)
        {
            if (itr_num == 0) cout << "MLEM iteration begin" << endl;

            // Normalize the reconstruction space;
            TDetEvent = thrust::reduce(thrust::device, cuda_recon_density, cuda_recon_density + X_Bins * Y_Bins * Z_Bins);
            thrust::transform(thrust::device, cuda_recon_density, cuda_recon_density + X_Bins * Y_Bins * Z_Bins, thrust::constant_iterator<double>(1.0 / TDetEvent), \
                              cuda_recon_density, thrust::multiplies<double>());

            // calculate the density amount of pixels swept by each event(fitol) and calculate the amount of 1/fitol for each voxel
            One_Batch_Integer_SysLis_Each_Event<<< MLEM_grid_dim, MLEM_block_dim, sizeof(double) * MLEM_block_dim >>> (len_effective, cuda_system_list, \
                                                                                                                       cuda_integr_event_index, cuda_recon_density, \
                                                                                                                       cuda_amount_divisor);
            cudaDeviceSynchronize();
            
            // iteration process
            thrust::transform(thrust::device, cuda_recon_density, cuda_recon_density + X_Bins * Y_Bins * Z_Bins, cuda_amount_divisor, cuda_recon_density, \
                              thrust::multiplies<double>());

            if (k != MLEM_Itr - 1)
            {
                CHECK(cudaMemset(cuda_amount_divisor, 0, X_Bins* Y_Bins* Z_Bins * sizeof(double)));
            }
            
            itr_num = k + 1;
            if (itr_num % MLEM_Display == 0)
            {
                CHECK(cudaMemcpy(recon_density, cuda_recon_density, X_Bins*Y_Bins*Z_Bins*sizeof(double), cudaMemcpyDeviceToHost));
                int mid_itr_num = itr_num;
                string mid_num;
                stringstream ss;
                ss<<mid_itr_num;
                ss>>mid_num;

                std::ofstream outList_mid;
                outList_mid.open(mid_num + "MLEM_back_projection.txt",ios::out);
                if(!outList_mid) cout<<"Open the file failure..\n"<<endl;

                for (int k_k = 0; k_k < Z_Bins; k_k++)
                {
                    for (int j_j = 0; j_j < Y_Bins; j_j++)
                    {
                        for (int i_i = 0; i_i < X_Bins; i_i++)
                        {
                            outList_mid<<recon_density[k_k * X_Bins * Y_Bins + j_j * X_Bins + i_i] << " ";
                        }
                        outList_mid<<endl;
                    }
                    outList_mid<<endl;
                }
                outList_mid.close();
            }
            cout << "MLEM: " << itr_num << " finished" << endl;
            if (itr_num == MLEM_Itr)
            {
                cout << "MLEM iteration completed" << endl;
            }
        }
        CHECK(cudaFree(cuda_system_list));
    }
    else  // multipal MLEM iteration batch mode
    {
        const int MLEM_block_dim = 128;
        int MLEM_grid_dim;
        if (amount_per_batch % MLEM_block_dim == 0) MLEM_grid_dim = int(amount_per_batch / MLEM_block_dim);
        else MLEM_grid_dim = int(amount_per_batch / MLEM_block_dim) + 1;

        for (int k = 0; k < MLEM_Itr; k++)
        {
            if (itr_num == 0) cout << "MLEM iteration begin" << endl;

            // Normalize the reconstruction space;
            TDetEvent = thrust::reduce(thrust::device, cuda_recon_density, cuda_recon_density + X_Bins * Y_Bins * Z_Bins);
            thrust::transform(thrust::device, cuda_recon_density, cuda_recon_density + X_Bins * Y_Bins * Z_Bins, thrust::constant_iterator<double>(1.0 / TDetEvent), \
                              cuda_recon_density, thrust::multiplies<double>());

            // calculate the density amount of pixels swept by each event(fitol) and calculate the amount of 1/fitol for each voxel
            for (int batch_index = 0; batch_index < iteration_batch_num; batch_index++)
            {   
                int* cuda_system_list;
                if (batch_index != (iteration_batch_num - 1))
                {
                    CHECK(cudaMalloc(&cuda_system_list, (integr_event_index[(batch_index + 1) * amount_per_batch] - integr_event_index[batch_index * amount_per_batch]) * \
                                     sizeof(int)));
                    CHECK(cudaMemcpy(cuda_system_list, &system_list[integr_event_index[batch_index*amount_per_batch]], (integr_event_index[(batch_index + 1) * \
                                     amount_per_batch] - integr_event_index[batch_index * amount_per_batch]) * sizeof(int), cudaMemcpyHostToDevice));
                }
                else
                {
                    CHECK(cudaMalloc(&cuda_system_list, (integr_event_index[len_effective] - integr_event_index[batch_index * amount_per_batch]) * sizeof(int)));
                    CHECK(cudaMemcpy(cuda_system_list, &system_list[integr_event_index[batch_index*amount_per_batch]], (integr_event_index[len_effective] - \
                                     integr_event_index[batch_index * amount_per_batch]) * sizeof(int), cudaMemcpyHostToDevice));
                }
                
                Multi_Batch_Integer_SysLis_Each_Event<<< MLEM_grid_dim, MLEM_block_dim, sizeof(double) * MLEM_block_dim >>> (len_effective, cuda_system_list, \
                                                                                                                             cuda_integr_event_index, cuda_recon_density, \
                                                                                                                             cuda_amount_divisor, batch_index, \
                                                                                                                             amount_per_batch);
                
                cudaDeviceSynchronize();
                CHECK(cudaFree(cuda_system_list));
            }

            // iteration process
            thrust::transform(thrust::device, cuda_recon_density, cuda_recon_density + X_Bins * Y_Bins * Z_Bins, cuda_amount_divisor, cuda_recon_density, \
                              thrust::multiplies<double>());

            if (k != MLEM_Itr - 1)
            {
                CHECK(cudaMemset(cuda_amount_divisor, 0, X_Bins* Y_Bins* Z_Bins * sizeof(double)));
            }
            
            itr_num = k + 1;
            if (itr_num % MLEM_Display == 0)
            {
                CHECK(cudaMemcpy(recon_density, cuda_recon_density, X_Bins*Y_Bins*Z_Bins*sizeof(double), cudaMemcpyDeviceToHost));
                int mid_itr_num = itr_num;
                string mid_num;
                stringstream ss;
                ss<<mid_itr_num;
                ss>>mid_num;

                std::ofstream outList_mid;
                outList_mid.open(mid_num + "MLEM_back_projection.txt",ios::out);
                if(!outList_mid) cout<<"Open the file failure..\n"<<endl;

                for (int k_k = 0; k_k < Z_Bins; k_k++)
                {
                    for (int j_j = 0; j_j < Y_Bins; j_j++)
                    {
                        for (int i_i = 0; i_i < X_Bins; i_i++)
                        {
                            outList_mid<<recon_density[k_k * X_Bins * Y_Bins + j_j * X_Bins + i_i] << " ";
                        }
                        outList_mid<<endl;
                    }
                    outList_mid<<endl;
                }
                outList_mid.close();
            }
            cout << "MLEM: " << itr_num << " finished" << endl;
            if (itr_num == MLEM_Itr)
            {
                cout << "MLEM iteration completed" << endl;
            }
        }
    }

    
    if (MLEM_Itr != 0) cout << "MLEM iteration is OK ! ! !" << endl;
    else cout << "No MLEM iteration" << endl;
    Print_Time_Complete(time_start, clock());
    cout << "**************************************************" << endl << endl;

    // Check cuda kernal function
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    // ********************* End of MLEM itreation********************** //


    // write out into file
    if (MLEM_Itr != 0)
    {
        CHECK(cudaMemcpy(recon_density, cuda_recon_density, X_Bins*Y_Bins*Z_Bins*sizeof(double), cudaMemcpyDeviceToHost));
        std::ofstream outList_1;
        outList_1.open("MLEM_back_projection.txt", ios::out);
        if (!outList_1) cout << "Open the file failure..\n" << endl;

        for (int k = 0; k < Z_Bins; k++)
        {
            for (int j = 0; j < Y_Bins; j++)
            {
                for (int i = 0; i < X_Bins; i++)
                {
                    outList_1 << recon_density[k * X_Bins * Y_Bins + j * X_Bins + i] << " ";
                }
                outList_1 << endl;
            }
            outList_1 << endl;
        }
        outList_1.close();
        cout << "MLEM Output file is OK ! ! !" << endl;
        Print_Time_Complete(time_start, clock());
        cout << "**************************************************" << endl << endl;
    }

    // Finally, destroy these three memory
    CHECK(cudaFree(cuda_recon_density));
    CHECK(cudaFree(cuda_integr_event_index));
    CHECK(cudaFree(cuda_amount_divisor));

    cudaDeviceReset();
    cout << "program finished!!!"<< endl;
    cout << "| ->_->           |" << endl;
    cout << "|      ->_<-      |" << endl;
    cout << "|           $-_-$ |" << endl;
    Print_Time_Complete(time_start, clock());
    return 0;
}
