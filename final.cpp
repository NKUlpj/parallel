#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include "time.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <arm_neon.h>
#include <omp.h>


using namespace std;

#define BLOCK_SIZE 16
__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

__global__ void Mat_sub_and_redu(float *A, float *B, float *d_local_sum,int m, int n) {
    float sum = 0;
    int my_ij = blockDim.x * blockIdx.x + threadIdx.x;
    if (blockIdx.x < m && threadIdx.x < n) {
        sum += abs(A[my_ij] - B[my_ij]);
        __syncthreads();
    } 
    if (threadIdx.x == 0)
	{
		d_local_sum[blockIdx.x] = sum;
	}   
} 

typedef unsigned int uint;
const int maxN = 1e4 +  10; // 最大样本数
const int maxF = 10;  // 最大特征数
float affinity_matrix[maxN][maxN];


template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
    // 初始化索引向量
    vector<size_t> idx(v.size());
    //使用iota对向量赋0~？的连续值
    iota(idx.begin(), idx.end(), 0);
    // 通过比较v的值对索引idx进行排序
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}


void labelPropagation(
        float Mat_Label[maxF][2],
        float Mat_UnLabel[maxN][2],
        int labels[maxN],
        int unlabel_data_labels[maxN],
        int num_label_samples,
        int num_unlabel_samples,
        string& kernel_type,
        float rbf_sigma,
        int knn_num_neighbors,
        int max_iter,
        float tol){
    int num_samples =  num_label_samples + num_unlabel_samples;
    set<int> labels_list;
    for(int i = 0; i < num_label_samples; i ++){
        labels_list.insert(labels[i]);
    }
    int num_classes = labels_list.size();
    float MatX[num_samples][2];
    memset(MatX,0,sizeof MatX);
    //拼接
    for(int i = 0; i < num_label_samples; i ++){
        for(int j = 0; j < 2; j ++){
            MatX[i][j] = Mat_Label[i][j];
        }
    }
    //#pragma omp parallel for num_threads(thread_count)
    for(int i = num_label_samples; i < num_samples; i ++){
        for(int j =0; j < 2; j ++){
            MatX[i][j] = Mat_UnLabel[i - num_label_samples][j];
        }
    }

    // 预处理，one-hot编码处理 标签
    //clamp 是原始数据的标签，不会被影响
    float clamp_data_label[num_label_samples][num_classes];
    memset(clamp_data_label,0,sizeof clamp_data_label);
    for(int i = 0; i < num_label_samples; i ++){
        clamp_data_label[i][labels[i]] = 1;
    }

    // 初始化原始的label矩阵，初始值为-1，不存在的标签。
    float label_function[num_samples][num_classes];
    memset(label_function,0,sizeof label_function);
    memcpy(label_function,clamp_data_label,sizeof clamp_data_label);

    // #pragma omp parallel for num_threads(thread_count)
    for(int i = num_label_samples; i < num_samples; i ++){
        for(int j = 0; j < num_classes; j ++){
            label_function[i][j] = -1;
        }
    }

    // 构建 n * n的邻接矩阵
    memset(affinity_matrix,0,sizeof affinity_matrix);
    if(kernel_type == "rbf"){
        cout << "todo" << endl;
        exit(1);
    } else{
        #pragma omp parallel for num_threads(thread_count)
        for(int i = 0; i < num_samples; i ++){
            // 计算每个点的邻居，为了画图方便，测试案例使用的是二维数据，此处没有使用SIMD加速
            vector<float> squaredDist(num_samples);
            // cache 优化
            float MatX_T[2][num_samples];
            for(int ti = 0; ti < 2; ti ++)
                for(int tj = 0; tj < num_samples; tj ++){
                    MatX_T[ti][tj] = MatX[tj][ti];
                }

            float32x4_t ta0,ta1,tb0,tb1;
            for(int l = num_samples - 4; l >= 0; l -= 4){
                ta0 = vld1q_f32(MatX_T[0] + l);
                ta1 = vld1q_f32(MatX_T[1] + l);

                tb0 = vdupq_n_f32(MatX[i][0]);
                tb1 = vdupq_n_f32(MatX[i][1]);

                // ta - tb
                ta0  = vsubq_f32(ta0,tb0);
                ta1  = vsubq_f32(ta1,tb1);

                // ta ^2
                ta0 = vmulq_f32(ta0,ta0);
                ta1 = vmulq_f32(ta1,ta1);

                ta0 = vaddq_f32(ta0,ta1);
                squaredDist[l + 0] = vgetq_lane_f32(ta0,0);
                squaredDist[l + 1] = vgetq_lane_f32(ta0,1);
                squaredDist[l + 2] = vgetq_lane_f32(ta0,2);
                squaredDist[l + 3] = vgetq_lane_f32(ta0,3);
            }
            // last
            for(int l = num_samples % 4 - 1; l >= 0; l --){
                float x = MatX_T[0][l] - MatX[i][0];
                float y = MatX_T[1][l] - MatX[i][1];
                squaredDist[l] = x * x + y * y;
            }

            // 取top k
            auto sortedDistIndices = sort_indexes(squaredDist);
            if(knn_num_neighbors > sortedDistIndices.size()){
                knn_num_neighbors = sortedDistIndices.size();
            }
            for(int t = 0; t < knn_num_neighbors;t ++){
                int ner_idx = sortedDistIndices[t];
                affinity_matrix[i][ner_idx] = 1.0 / knn_num_neighbors;
            }
        }
    }
    // 开始迭代
    int iter = 0;
    float pre_label_function[num_samples][num_classes];
    memset(pre_label_function,0.0f,sizeof  pre_label_function);

    float pre_changed = 0;
    float cur_changed = 0;
#pragma omp parallel for num_threads(4) reduction(+:cur_changed)
    for(int i = 0; i < num_samples; i ++){
        float32x4_t t_pre,t_cur;
        float32x4_t res_change = vmovq_n_f32(0);
        for(int j = num_classes - 4; j >= 0; j -=4){
            t_pre = vld1q_f32(pre_label_function[i] + j);
            t_cur = vld1q_f32(label_function[i] + j);
            t_pre = vsubq_f32(t_pre,t_cur);
            res_change = vaddq_f32(res_change, vabsq_f32(t_pre));
        }
        float32x2_t res_change_low = vget_low_f32(res_change);
        float32x2_t res_change_high = vget_high_f32(res_change);
        res_change_low = vpadd_f32(res_change_low, res_change_high);
        cur_changed += (float)vpadds_f32(res_change_low);
        // 计算剩余的四个元素
        for(int j = num_classes % 4 - 1; j >= 0; --j){
            cur_changed+= abs(pre_label_function[i][j] - label_function[i][j]);
        }
    }
    // 准备数组
    int m = num_samples, n = num_samples, k = num_classes;

    // ========== local ================
    float *h_a;
    cudaMallocHost((void **) &h_a, sizeof(float) * num_samples * num_samples);
    // ========== GPU ================
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float) * m * n);
    cudaMalloc((void **) &d_b, sizeof(float) * n * k);
    cudaMalloc((void **) &d_c, sizeof(float) * m * k);


    float *d_pre; // gpu 前一个label
    float *d_local_sum = nullptr; // gpu 局部和
    float *h_local_sum = nullptr; // cpu 局部和

    cudaMalloc((void **) &d_pre, sizeof(float) * m * k);
    cudaMalloc((void**)&d_local_sum, m * sizeof(float));
    h_local_sum = (float*)malloc(m * sizeof(float));


    // ========== h_a ================
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_samples; ++j) {
            h_a[i * num_samples + j] = affinity_matrix[i][j];
        }
    }
    while(iter < max_iter && abs(pre_changed - cur_changed) > tol){
        iter += 1;
        memcpy(pre_label_function,label_function,sizeof label_function);
        cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, label_function, sizeof(float)*n*k, cudaMemcpyHostToDevice);
        unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
        cudaMemcpy(label_function, d_c, sizeof(float)* m * k, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //保证原始数据不会跑偏
        memcpy(label_function,clamp_data_label,sizeof clamp_data_label);

        pre_changed = cur_changed;
        cur_changed = 0;

        cudaMemcpy(d_pre, pre_label_function, sizeof(float)* n * k, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, label_function, sizeof(float) * n * k, cudaMemcpyHostToDevice);

        Mat_sub_and_redu<<<n, k>>>(d_pre,d_c, d_local_sum,n, k);
        cudaDeviceSynchronize();
        cudaMemcpy(h_local_sum, d_local_sum, n * sizeof(float),cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++)
        {
            cur_changed += h_local_sum[i];
        }


    }
    for(int i = 0; i < num_unlabel_samples; i ++){
        vector<float> v;
        for(int j = 0; j < num_classes; j ++){
            v.push_back(label_function[i + num_label_samples][j]);
        }
        auto biggest = max_element(begin(v), end(v));
        int pos = distance(begin(v), biggest);
        unlabel_data_labels[i] = pos;
    }
}



//读取二维数组
int read_test_data(string& filename,float dataSets[][2]){
    ifstream csv_data(filename, ios::in);
    string line;
    if (!csv_data.is_open())
    {
        cout << "Error: opening file fail" << endl;
        exit(1);
    }
    istringstream sin;         //将整行字符串line读入到字符串istringstream中
    int i = 0;
    int j = 0;
    string word;
    // 读取数据
    while (getline(csv_data, line))
    {
        sin.clear();
        sin.str(line);
        while (getline(sin, word, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
        {
            dataSets[i][j] = stof(word);
            j ++;
        }
        i ++;
        j = 0;
    }
    csv_data.close();
    return i;
}

//写结果
void write_test_data(string &filename,int labels[maxN],int unlabel_num){
    ofstream outFile;
    outFile.open(filename, ios::out | ios::trunc);
    for(int i = 0; i < unlabel_num; i ++){
        outFile << to_string(labels[i]);
        if(i != unlabel_num - 1){
            outFile << ",";
        }
    }
    outFile.close();
}

int main() {
    float Mat_Unlabel[maxN][2];
    float Mat_Label[maxF][2];
    int unlabel_data_labels[maxN];
    //8个特征
    int labels[8] = {0,1,2,3,4,5,6,7};
    int test_scale[11] = {128,256,512,1024,2048,3072,4096,5120,6144,7168,8192};
    //int test_scale[5] = {128,256,512,1024,2048};
    string kernel_type = "knn";
    for(auto num: test_scale){
        string mat_unlabel_csv = "Mat_Unlabel_" + to_string(num) + ".csv";
        string mat_label_csv = "Mat_Label_" + to_string(num) + ".csv";
        int num_un_label = read_test_data(mat_unlabel_csv, Mat_Unlabel);
        int num_label = read_test_data(mat_label_csv, Mat_Label);
        //cout << "read label data:" << num_label << "unlabel data:" << num_un_label << endl;
        auto ls = clock();
        labelPropagation(
                Mat_Label,
                Mat_Unlabel,
                labels,
                unlabel_data_labels,
                num_label,
                num_un_label,
                kernel_type,
                1.5,
                10,
                2000,
                0.001);
        auto le = clock();
        cout << "数据量为i:"<< num << ";运行时长:" << (double )(le - ls) / CLOCKS_PER_SEC << endl;
        string labels_csv = "test_data_res_" + to_string(num) + ".csv";
        write_test_data(labels_csv,unlabel_data_labels,num);
    }
    return 0;
}
