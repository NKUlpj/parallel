# parallel
* Base.cpp:基础代码
* SIMD.cpp:neon
* SSE.cpp:SSE指令集
* scripy.py:生成测试数据和画图脚本
* openMP_v1.cpp: openMP并行
* openMP_v2.cpp: 考虑线程创建与销毁
* cuda_v1.cu: 使用CUDA实现任意形状的矩阵乘法和带有规约操作的矩阵减法


> SSE.cpp编译时，在makefile添加 `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")`
<<<<<<< HEAD

=======
> 
>>>>>>> a30c517109f2b321bb60033013f084b79d470b48
> openMP编译时，请在makefile添加
```
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
```
使用g++编译，不要使用clang。

> cuda编译命令`nvcc cuda_v1.cu -lcublas -lcusparse -o cudav1`


