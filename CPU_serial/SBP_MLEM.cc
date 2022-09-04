#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <list>
#include <vector>
#include <time.h>
#include <assert.h>
#include <cstring>
#include <sstream>

using namespace std;

#define Electron_Mass 510.99       // keV
#define Pi 3.14159              // circumference
#define X_POS 0
#define Y_POS 1
#define Z_POS 2
//**********************************//
#define Input_File "True_CZT478.txt"    // input file
#define dtheta 0.03             //rad
#define Gamma_Energy 478        // keV (the energy of incident photon)
#define Denergy 3               // keV (the energy range for recorded events)
#define X_Bins 100              // bins
#define Y_Bins 100              // bins
#define Z_Bins 100              // bins
#define xmin -100               // mm
#define xmax 100                // mm
#define ymin -100               // mm
#define ymax 100                // mm
#define zmin -100               // mm
#define zmax 100                // mm   
#define Distance_Filter 10       // mm  (0 means no Distance filter)
//*********************************//
#define Events_Display 1000
#define MLEM 40
#define MLEM_Display 4

// define matrix class
class Density
{
    public:
    Density();
    Density(int z_num, int y_num, int x_num, int event_num = 1); 
    ~Density() {};
    double &data(int z_index, int y_index, int x_index, int event_index = 0);
    double &at(int index);
    void reset();

    double *count;
    int X_num, Y_num, Z_num, Event_num;
};
Density::Density()
{}
Density::Density(int z_num, int y_num, int x_num, int event_num)
{
    X_num = x_num;
    Y_num = y_num;
    Z_num = z_num;
    Event_num = event_num;
    int Total_num = Event_num*X_num*Y_num*Z_num;

    count = new double[Total_num]();
}
double &Density::data(int z_index, int y_index, int x_index, int event_index)
{
    int index;
    if ((x_index < X_num) & (y_index < Y_num) & (z_index < Z_num) & (event_index < Event_num))
    {
        index = event_index*X_num*Y_num*Z_num + z_index*X_num*Y_num + y_index*X_num + x_index;
        return count[index];
    }
    else abort();
}
double &Density::at(int index)
{
    if (index < X_num*Y_num*Z_num) return count[index];
    else abort();
}
void Density::reset()
{
    int N = Event_num*X_num*Y_num*Z_num;
    memset(count, 0, N*sizeof(double));
}

// used functions
// ********************************************************************************//
double GetLength(double z1, double y1, double x1, double z2, double y2, double x2)
{
    double len = sqrt(pow((x1-x2),2)+pow((y1-y2),2)+pow((z1-z2),2));
    return len;
}
void Print_Time_Complete( double clock_start , double clock_end , bool fin = 0 )
{
    double total_time =  (clock_end - clock_start)/(CLOCKS_PER_SEC);
    unsigned minutes = total_time/60;
    double seconds = total_time - minutes*60;

    string pref = " -- Time Taken =";
    string post = "            \n -- Done\n";

    if ( fin == 1 ){
        pref = "\nReconstruction Complete \nTotal Runtime =";
        post = "            \n\n";
    }

    if ( minutes > 0 ) printf( "%s %4u minutes and %4.2f seconds %s" , pref.c_str() , minutes , seconds , post.c_str() );
    else               printf( "%s %4.2f seconds %s" , pref.c_str() , seconds , post.c_str() );
}
// ********************************************************************************//

// define Compton event class
class Compton_event
{
    public:
    Compton_event();
    Compton_event(double x1, double y1, double z1, double x2, double y2, double z2, double e1, double e2); 
    ~Compton_event() {};
    bool Check(void);
    void GetComptonAngle(void);
    bool Back_projection(double z, double y, double x);
    
    double x_scatter, y_scatter, z_scatter, e_scatter;
    double x_absorb, y_absorb, z_absorb, e_absorb;
    double cos_com_sca_angle;
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
}
bool Compton_event::Check()
{
    double E_sca_max = (2*Gamma_Energy*Gamma_Energy)/(Electron_Mass + 2*Gamma_Energy);
    return (e_scatter < E_sca_max);
}
void Compton_event::GetComptonAngle()
{
    double cos_angle = 1 - (Electron_Mass*e_scatter)/(e_scatter*e_absorb + pow(e_absorb, 2));
    cos_com_sca_angle = cos_angle;
    //cout<<"康普顿角： "<<cos_com_sca_angle<<endl;
}
bool Compton_event::Back_projection(double z, double y, double x)
{
    bool pDensity;

    double len1 = GetLength(x, y, z, x_scatter, y_scatter, z_scatter);
    double len2 = GetLength(x_scatter, y_scatter, z_scatter, x_absorb, y_absorb, z_absorb);

    double Prod = (x - x_scatter) * (x_scatter - x_absorb) + (y - y_scatter) * (y_scatter - y_absorb) + (z - z_scatter) * (z_scatter - z_absorb);
    double r_comptonAngle = Prod / (len1 * len2);
    double discT = abs(acos(r_comptonAngle) - acos(cos_com_sca_angle));

    if (discT < dtheta) pDensity = true;
    else pDensity = false;

    return pDensity;
};

int main()
{
    ifstream infile;
    infile.open(Input_File, ios::in);
    if(!infile) cout<<"ifstream failure ! ! !"<<endl;
    else cout<<"ifstream good ! ! !"<<endl;

    double time_start = clock();
    double rec_x_voxel = (xmax - xmin)/double(X_Bins); 
    double rec_y_voxel = (ymax - ymin)/double(Y_Bins);
    double rec_z_voxel = (zmax - zmin)/double(Z_Bins);

    double x1, x2, y1, y2, z1 ,z2, e1, e2;
    int count_event = -1;  // when count_event is equl to 1, it means one slice of elements will be included in Recon_Space or System_Matrix,
                           // as for current situation, the initialization of count_event should be -1.
                           // and this is due to the last line of the input file is empty .

    std::list<Compton_event*> Compton_event_Set;
    while (!infile.eof())
    {
        infile>>x1>>y1>>z1>>x2>>y2>>z2>>e1>>e2;
        if (((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2)) < pow(Distance_Filter, 2)) continue;  // distance filter
        if (abs((e1 + e2) - Gamma_Energy) > Denergy) continue;                                              // energy filter                                             

        Compton_event* event= new Compton_event(x1, y1, z1, x2, y2, z2, e1, e2);
        if (!(event->Check()))                                                                              // Compton edge filter
        {
            continue;
        }                                                                 
        Compton_event_Set.push_back(event);
        count_event++;
    }
    std::list<Compton_event*>::iterator it1 = Compton_event_Set.end();
    it1--;
    it1 = Compton_event_Set.erase (it1);  // delete the last useless elements
    // End of importing file

    // print out data
    const int len = Compton_event_Set.size();
    cout<<"Compton Event number: "<<len<<endl;

    Density Recon_Space(Z_Bins, Y_Bins, X_Bins);
    vector<vector<int> > System_Matrix_List;
    int system_list_index;

    // fill the density matrix in two ways (SBP and Ray_Tracing)
    std::list<Compton_event*>::iterator itr;
    int event_id = 0;

    for(itr = Compton_event_Set.begin(); itr != Compton_event_Set.end(); itr++)
    {
        (*itr)->GetComptonAngle();
        vector<int> system_list;
        for (int z_set = 0; z_set < Z_Bins; z_set++)
        {
            for (int y_set = 0; y_set < Y_Bins; y_set++)
            {
                for (int x_set = 0; x_set < X_Bins; x_set++)
                {
                    double z_cor = zmin + z_set*rec_z_voxel + 0.5*rec_z_voxel;
                    double y_cor = ymin + y_set*rec_y_voxel + 0.5*rec_y_voxel;
                    double x_cor = xmin + x_set*rec_x_voxel + 0.5*rec_x_voxel;

                    bool pass = (*itr)->Back_projection(z_cor, y_cor, x_cor);
                    if(pass)
                    {
                        Recon_Space.data(z_set, y_set, x_set)++; 
                        system_list_index = z_set*X_Bins*Y_Bins + y_set*X_Bins + x_set;
                        system_list.push_back(system_list_index);
                    }
                }
            }
        }
        if (system_list.size() != 0)
        {
            event_id++;
            System_Matrix_List.push_back(system_list);
        }
        if (event_id%Events_Display == 0) cout<<event_id<<endl;
    }
    const int eff_len = System_Matrix_List.size();
    cout<<"Effective event: "<<eff_len<<endl;;
    cout<<"end of back projection"<<endl;
    Print_Time_Complete( time_start , clock() );

    // end of Simple Back Projection iteration
    // write out into file
    std::ofstream outList;
	outList.open("Simple_back_projection.txt",ios::out);
    if(!outList) cout<<"Open the file failure..\n"<<endl;

    for (int k = 0; k < Z_Bins; k++)
    {
        for (int j = 0; j < Y_Bins; j++)
        {
            for (int i = 0; i < X_Bins; i++)
            {
                outList<<Recon_Space.data(k, j, i)<<" ";
            }
            outList<<endl;
        }
        outList<<endl;
    }
    outList.close();
    cout<<"Output file OK ! ! !"<<endl;
    Print_Time_Complete( time_start , clock() );
    cout<<"**************************************************"<<endl;

    /************************MLEM itreation**************************************/
    int itr_num = 0;
    double TDetEvent;
    for (int k = 0; k < MLEM; k++)
    {
        if (itr_num == 0) cout<<"MLEM iteration begin"<<endl;
        vector<double> f_tol_event(eff_len);
        vector<double> amount_divisor(X_Bins*Y_Bins*Z_Bins);
        TDetEvent = 0.0;
        
        // calculate the total density of reconstruction space
        for (int IsZ = 0; IsZ < Z_Bins; IsZ++)
        {
            for (int IsY = 0; IsY < Y_Bins; IsY++)
            {
                for (int IsX = 0; IsX < X_Bins; IsX++)
                {
                    TDetEvent += Recon_Space.data(IsZ, IsY, IsX);
                }
            }
        }

        // normalize of reconstruction space
        for (int IsZ = 0; IsZ < Z_Bins; IsZ++)
        {
            for (int IsY = 0; IsY < Y_Bins; IsY++)
            {
                for (int IsX = 0; IsX < X_Bins; IsX++)
                {
                    Recon_Space.data(IsZ, IsY, IsX) = Recon_Space.data(IsZ, IsY, IsX) / TDetEvent;
                }
            }
        }
        
        // calculate the density amount of pixels swept by each event
        for (int i = 0; i < eff_len; i++)
        {
            for (int j = 0; j < (System_Matrix_List[i]).size(); j++ )
            {
                f_tol_event[i] += Recon_Space.at((System_Matrix_List[i][j]));
            }
        }

        // calculate the 1/fitol for each event
        for (int i = 0; i < eff_len; i++)
        {
            for (int j = 0; j < (System_Matrix_List[i]).size(); j++ )
            {
                amount_divisor[System_Matrix_List[i][j]] += (1.0/(f_tol_event[i]));
            }
        }

        // iteration process
        for (int IsZ_1 = 0; IsZ_1 < Z_Bins; IsZ_1++)
        {
            for (int IsY_1 = 0; IsY_1 < Y_Bins; IsY_1++)
            {
                for (int IsX_1 = 0; IsX_1 < X_Bins; IsX_1++)
                {
                    Recon_Space.data(IsZ_1, IsY_1, IsX_1) = Recon_Space.data(IsZ_1, IsY_1, IsX_1)*\
                    amount_divisor[(IsZ_1*X_Bins*Y_Bins + IsY_1*X_Bins + IsX_1)];
                }
            }
        }
        
        itr_num = k + 1;
        if (itr_num % MLEM_Display == 0 && itr_num >= 10)
        {
            int mid_itr_num = itr_num;
            string mid_num;
            stringstream ss;
            ss<<mid_itr_num;
            ss>>mid_num;

            std::ofstream outList_mid;
	        outList_mid.open(mid_num + "MLEM_projection.txt",ios::out);
            if(!outList_mid) cout<<"Open the file failure..\n"<<endl;

            for (int k = 0; k < Z_Bins; k++)
            {
                for (int j = 0; j < Y_Bins; j++)
                {
                    for (int i = 0; i < X_Bins; i++)
                    {
                        outList_mid<<Recon_Space.data(k, j, i)<<" ";
                    }
                    outList_mid<<endl;
                }
                outList_mid<<endl;
            }
            outList_mid.close();
        }
        cout<<"MLEM: "<<itr_num<<" finished"<<endl;
        if (itr_num == MLEM)
        {
            cout << "MLEM iteration completed" << endl; 
            cout << "**************************************************" << endl; 
        }
    }
    cout<<"MLEM iteration is OK ! ! !"<<endl;
    Print_Time_Complete( time_start , clock() );


    // end of MLEM iteration
    // write out into file
    std::ofstream outList_1;
	outList_1.open("MLEM_back_projection.txt",ios::out);
    if(!outList_1) cout<<"Open the file failure..\n"<<endl;

    for (int k = 0; k < Z_Bins; k++)
    {
        for (int j = 0; j < Y_Bins; j++)
        {
            for (int i = 0; i < X_Bins; i++)
            {
                outList_1<<Recon_Space.data(k, j, i)<<" ";
            }
            outList_1<<endl;
        }
        outList_1<<endl;
    }
    outList_1.close();
    cout<<"Output file OK ! ! !";

    return 0;
}