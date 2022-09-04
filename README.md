# Compton-Camera-Imaging-Reconstruction
摘要：
当前文件夹下包含两个文件夹“CPU_serial/“和“GPU_parallel/“，
以及康普顿重建数据参考输入文件“True_CZT478.txt“和重建结果可视化脚本“ThreeD_compton_image.m“
“CPU_serial/“和“GPU_parallel/“包含的程序功能同为康普顿相机图像重建，重建结果经测试没有差异。
“CPU_serial/“包含程序为CPU上串行执行的康普顿相机重建程序，包含角度展宽的预先反投影以及MLEM迭代；
“GPU_parallel/“包含程序为GPU上并行执行的康普顿相机重建程序，角度展宽的预先反投影以及MLEM迭代过程都将并行执行。

硬件及环境要求：
"CPU_serial/“仅依赖于标准C++库及编译器，windows端可下载mingw "https://sourceforge.net/projects/mingw/"，并配置环境变量；
"GPU_parallel/“依赖于CUDA及相关环境，windows端可首先安装MS 2022社区版 "https://visualstudio.microsoft.com/zh-hans/"，接着更新显卡驱动并安装
合适版本的CUDA Toolkit "https://developer.nvidia.com/cuda-downloads"，并配置环境变量；
"GPU_parallel/“硬件要求计算能力高于6.0的显卡，可参考"https://developer.nvidia.com/zh-cn/cuda-gpus#compute" 以了解是否符合计算能力要求。

程序使用：
1. "CPU_serial/“
"CPU_serial/“文件夹下仅有一个.cc脚本，SBP_MLEM.cc即为重建程序；
SBP_MLEM.cc程序顶部为重建设置参数：
------------------------------------------------------------------------------------------
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
------------------------------------------------------------------------------------------
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
