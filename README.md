# parallel
## 第一次作业
* Base.cpp:基础代码
* SIMD.cpp:neon
* SSE.cpp:SSE指令集
* scripy.py:生成测试数据和画图脚本
* openMP_v1.cpp: openMP并行
* openMP_v2.cpp: 考虑线程创建与销毁


> SSE.cpp编译时，在makefile添加 `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")`
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


