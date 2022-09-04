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
### "CPU_serial/"
There is only one .cc script in the "CPU_serial/" folder, and SBP_MLEM.cc is the reconstruction program;<br>
The top of the SBP_MLEM.cc program sets the parameters for the rebuild:<br>

  #define Electron_Mass 510.99       // keV<br>
  #define Pi 3.14159              // circumference<br>
  #define X_POS 0<br>
  #define Y_POS 1<br>
  #define Z_POS 2<br>
  "//**********************************//"<br>
  "#define Input_File "True_CZT478.txt"    // input file"<br>
  "#define dtheta 0.03             //rad"<br>
  "#define Gamma_Energy 478        // keV (the energy of incident photon)"<br>
  "#define Denergy 3               // keV (the energy range for recorded events)"<br>
  "#define X_Bins 100              // bins"<br>
  "#define Y_Bins 100              // bins"<br>
  "#define Z_Bins 100              // bins"<br>
  "#define xmin -100               // mm"<br>
  "#define xmax 100                // mm"<br>
  "#define ymin -100               // mm"<br>
  "#define ymax 100                // mm"<br>
  "#define zmin -100               // mm"<br>
  "#define zmax 100                // mm"<br>
  "#define Distance_Filter 10       // mm  (0 means no Distance filter)"<br>
  "//*********************************//"<br>
  "#define Events_Display 1000"<br>
  "#define MLEM 40"<br>
  "#define MLEM_Display 4"<br>

其中，前四行为常量，可不做更改；
"Input_File"为输入文件的文件名，使用时需要放在本程序同一文件夹目录下；
"dtheta"为直接反投影的角度展宽量，单位为弧度制；
"Gamma_Energy"为入射光子能量，参考探测的目标光子能量进行修改；
"Denergy”为允许的入射光子能量展宽量，可根据探测设备能量分辨率进行修改；
接下来九行为重建空间参数，分别是X，Y，Z方向重建的bin数，以及X，Y，Z方向上重建空间包含的空间范围，实例中展现的重建空间为以原点为中心的三维正方体，
实际使用中可根据需要任意更改，二维成像同样支持，将任意方向的_Bins值更改为1后对应更改成像空间范围即可进行二维重建；
"Distance_Filter”为距离筛选量，筛选目标为散射和吸收作用发生位置的距离量，低于Distance_Filter的事件将不被考虑进行重建；
"Events_Display”仅用于直接反投影阶段监视重建的进度，本实例中的意义为每重建1000事件发送一次当前重建进度信息；
"MLEM”为MLEM迭代总次数，可根据需要任意设置；
"MLEM_Display”用于设置迭代一定次数后进行重建结果的输出，本实例中的意义为每迭代4次进行一次重建结果输出；
------------------------------------------------------------------------------------------
SBP_MLEM.cc输出结果为矩阵形式，该矩阵被写入.txt文件中，可由ThreeD_compton_image.m进行可视化；
SBP_MLEM.cc的参考输入文件格式为x1 y1 z1 x2 y2 z3 e1 e2；其中1表示散射事件，2表示吸收事件；
SBP_MLEM.cc程序运行为标准C++模式，编译+运行可执行文件。


2. "GPU_parallel/“
"GPU_parallel/“文件夹下有两个文件夹"include"和"src"以及一个文件"makefile"
其中"include"包含有重建程序所需头文件，"src"包含有重建程序执行的源文件，makefile为编译命令文件
使用时，依据需要修改"src/main_function.cu"文件，同SBP_MLEM.cc，本程序重建参数也位于程序顶部：
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
由于CPU程序和GPU程序功能完全一致，因此重建参数含义与SBP_MLEM.cc相同，这里仅对两者不一样的地方进行解释：
"Max_Ava_Video_Me"为计算机可用的最大显存大小，单位为MiB，可在计算机终端输入nvidia-smi指令查看当前计算机可用显存容量大小，使用时最好略小于可用显存大小；
"Compton_Ang_Filter"为GPU程序添加功能，即选择是否进行散射角度筛选，筛选范围为0-120度；
------------------------------------------------------------------------------------------
GPU程序的运行方式为：
1.GPU程序的编译依赖于makefile的命令，makefile需要依据实际情况修改一处参数设置：
------------------------------------------------------------------------------------------
NVFLAGS = -arch=compute_61 -code=sm_61
------------------------------------------------------------------------------------------
本实例参考1080 Ti显卡（计算能力 6.1），实际使用时需要设置为："NVFLAGS = -arch=compute_计算能力*10 -code=sm_计算能力*10"；
2.makefile修改后，在"GPU_parallel/“目录下进行编译，编译命令为"make"；
3.执行可执行程序。
GPU程序的输入和输出与SBP_MLEM.cc相同。

如有问题，请联系邬仁耀 qq:1243511701
