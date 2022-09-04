# Compton-Camera-Imaging-Reconstruction
## Abstract：
1. The current path contains two folders "CPU_serial/" and "GPU_parallel/";<br>
2. The Compton reconstruction data refer to the input file "True_CZT478.txt" and the reconstruction result visualization script "ThreeD_compton_image.m";<br>
3. The program functions included in "CPU_serial/" and "GPU_parallel/" are the same as Compton camera image reconstruction, and there is no difference in the reconstruction results after testing.<br>
4. "CPU_serial/" contains the program for the serial execution of the Compton camera reconstruction program on the CPU, including resolution-corrected pre-backprojection and LM-MLEM iterations;<br>
5. "GPU_parallel/" contains the program for the Compton camera reconstruction program to be executed in parallel on the GPU. The pre-backprojection for resolution correction and the MLEM iteration process will all be executed in parallel.<br>

## Hardware and Environmental Requirements:

1. "CPU_serial/" only depends on the standard C++ library and compiler, you can download mingw "https://sourceforge.net/projects/mingw/" on the windows side, and configure environment variables;<br>
2. "GPU_parallel/" depends on CUDA and related environments, the Windows side can first install the MS 2022 Community Edition"https://visualstudio.microsoft.com/zh-hans/", then update the graphics card driver and install the appropriate version of CUDA Toolkit "https://developer.nvidia.com/cuda-downloads", and configure environment variables;
3. "GPU_parallel/" hardware requires a graphics card with computing power higher than 6.0. You can refer to "https://developer.nvidia.com/zh-cn/cuda-gpus#compute" to see if it meets the computing power requirements.<br>

## Program use:
### CPU_serial/
#### usage
There is only one .cc script in the "CPU_serial/" folder, and SBP_MLEM.cc is the reconstruction program;<br>
The top of the SBP_MLEM.cc program sets the parameters for the rebuild:<br>
```C
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
```
1. Among them, the first five behaviors are constants and can not be changed;  
2. "Input_File" is the file name of the input file, which needs to be placed in the same folder of the program when using it;  
3. "dtheta" is the angle broadening of direct back projection, in radians;  
4. "Gamma_Energy" is the incident photon energy, modified with reference to the detected target photon energy;  
5. "Denergy" is the allowable spread of incident photon energy, which can be modified according to the energy resolution of the detection device;  
6. The next nine rows are the reconstructed space parameters, which are the number of bins reconstructed in the X, Y, and Z directions, and the spatial range contained in the reconstructed space in the X, Y, and Z directions. The reconstructed space shown in the example is a three-dimensional cube centered on the origin. In actual use, it can be arbitrarily changed as needed, and 2D imaging is also supported. After changing the Z_Bins value in any direction to 1, the imaging space range can be changed to perform 2D reconstruction;  
7. "Distance_Filter" is the distance filtering amount, and the filtering target is the distance amount where scattering and absorption occur. Events below Distance_Filter will not be considered for reconstruction;  
8. "Events_Display" is only used to monitor the progress of reconstruction in the direct back projection stage. The meaning in this example is to send the current reconstruction progress information every 1000 reconstruction events;
9. "MLEM" is the total number of MLEM iterations, which can be set arbitrarily as needed;
10. "MLEM_Display" is used to set the output of the reconstruction result after a certain number of iterations. The meaning in this example is to output the reconstruction result every 4 iterations;
#### supplement
1. The output of SBP_MLEM.cc is in the form of a matrix, which is written into a .txt file and can be visualized by "ThreeD_compton_image.m";  
2. The reference input file format of SBP_MLEM.cc is x1 y1 z1 x2 y2 z3 e1 e2; where 1 is a scattering event and 2 is an absorption event;  
3. The SBP_MLEM.cc program runs in standard C++ mode, compiling + running the executable.

### GPU_parallel/
#### usage
There are two folders "include" and "src" and a file "makefile" under the "GPU_parallel/" folder;  
Among them, "include" contains the header files required for the reconstruction program, "src" contains the source files executed by the reconstruction program, and makefile is the compilation command file;  
When using, modify the "src/main_function.cu" file as needed, same as SBP_MLEM.cc, the reconstruction parameters of this program are also located at the top of the program:

```C
------------------------------------------------------------------------------------------
// *** Input File *** //
#define Input_File "./True_CZT478.txt" 
//**********************************//

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
------------------------------------------------------------------------------------------
```
1. Since the functions of the CPU program and the GPU program are exactly the same, the meaning of the reconstruction parameters is the same as that of SBP_MLEM.cc. Only the differences between the two are explained here;  
2. "Max_Ava_Video_Me" is the maximum video memory size available to the computer, the unit is MiB, you can enter the nvidia-smi command in the computer terminal to check the current computer available video memory size, preferably slightly smaller than the available video memory size when using;
3. "Compton_Ang_Filter" adds a function to the GPU program, that is, choose whether to filter the scattering angle, and the filter range is 0-120 degrees;  
#### supplement
The compilation of the GPU program depends on the command of the makefile. The makefile needs to modify a parameter setting according to the actual situation:
```Shell
NVFLAGS = -arch=compute_61 -code=sm_61
```
1. This example refers to the 1080 Ti graphics card (computing capability 6.1). In actual use, it needs to be set to: "NVFLAGS = -arch=compute_computing-capability*10 -code=sm_computing-capability*10";  
2. After the makefile is modified, compile it in the "GPU_parallel/" directory, and the compile command is "make";  
3. Execute the executable program.
The inputs and outputs of the GPU program are the same as SBP_MLEM.cc.

## If you have any questions, please contact Wu Renyao  
qq:"    "1243511701  
email:" "wurenyao@nuaa.edu.cn
