#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include "time.h"
#include <immintrin.h>


using namespace std;


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
//数据是二维数据
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
    for(int i = num_label_samples; i < num_samples; i ++){
        for(int j = 0; j < num_classes; j ++){
            label_function[i][j] = -1;
        }
    }

    // 构建 n * n的邻接矩阵
    memset(affinity_matrix,0,sizeof affinity_matrix);
    if(kernel_type == "rbf"){
        // w_{i,j} = exp(-\frac{|xi - xj|^2}{\alpha^2})
        /*
         for(int i = 0; i < num_samples; i ++){
            for(int j = 0; j < num_samples; j ++){
                float module[1];
                for(int k = features - 4; k >= 0; k -=4){
                    ta = _mm_loadu_ps(MatX[i] + k);
                    tb = _mm_loadu_ps(MatX[j] + k);
                    ta = _mm_sub_ps(ta, tb);
                    res4 = _mm_add_ps(ta,ta);
                }

                res4 =  _mm_hadd_ps(res4,res4);
                res4 =  _mm_hadd_ps(res4,res4);
                _mm_store_ss(module,res4);
                for(int k = features % 4 - 1; k >= 0; k --){
                    module[0] += (MatX[i][k] - MatX[j][k]) * (MatX[i][k] - MatX[j][k])
                }
                affinity_matrix[i][j] = exp(-(module[0])/(rbf_sigma * rbf_sigma));
            }
        }
         * */
        cout << "todo" << endl;
        exit(1);
    } else{
        for(int i = 0; i < num_samples; i ++){
            // 计算每个点的邻居，为了画图方便，测试案例使用的是二维数据，此处没有使用SIMD加速
            vector<float> squaredDist(num_samples);
/*            for(int j = 0; j < num_samples; j ++){
                for(int k = 0; k < 2; k ++){
                    float residual = MatX[j][k] - MatX[i][k];
                    squaredDist[j] += residual * residual;
                }
            }*/
            // cache 优化
            float MatX_T[2][num_samples];
            for(int ti = 0; ti < 2; ti ++)
                for(int tj = 0; tj < num_samples; tj ++){
                    MatX_T[ti][tj] = MatX[tj][ti];
                }

            __m128 ta0,ta1,tb0,tb1;
            for(int l = num_samples - 4; l >= 0; l -= 4){
                ta0 = _mm_loadu_ps(MatX_T[0] + l);
                ta1 = _mm_loadu_ps(MatX_T[1] + l);

                tb0 = _mm_set_ps(MatX[i][0],MatX[i][0],MatX[i][0],MatX[i][0]);
                tb1 = _mm_set_ps(MatX[i][1],MatX[i][1],MatX[i][1],MatX[i][1]);

                // ta - tb
                ta0  = _mm_sub_ps(ta0,tb0);
                ta1  = _mm_sub_ps(ta1,tb1);

                // ta ^2
                ta0 = _mm_mul_ps(ta0,ta0);
                ta1 = _mm_mul_ps(ta1,ta1);

                ta0 = _mm_add_ps(ta0,ta1);
                float squared[4];
                _mm_store_ps(squared,ta0);
                squaredDist[l + 0] = squared[0];
                squaredDist[l + 1] = squared[1];
                squaredDist[l + 2] = squared[2];
                squaredDist[l + 3] = squared[3];

            }
            // last
            for(int l = num_samples % 4 - 1; l >= 0; l --){
                int x = MatX_T[0][l] - MatX[i][0];
                int y = MatX_T[1][l] - MatX[i][1];
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
    float changed[2] = {0,0};
    for(int i = 0; i < num_samples; i ++){
        __m128 t_pre,t_cur;
        __m128 res_change = _mm_setzero_ps();
        for(int j = num_classes - 4; j >= 0; j -=4){
            t_pre = _mm_loadu_ps(pre_label_function[i] + j);
            t_cur = _mm_loadu_ps(label_function[i] + j);
            t_pre = _mm_add_ps(t_pre,t_cur);
            res_change = _mm_add_ps(res_change, _mm_abs_epi16(t_pre));
        }
        res_change = _mm_hadd_ps(res_change,res_change);
        res_change = _mm_hadd_ps(res_change,res_change);
        _mm_store_ss(changed + 1, res_change);

        // 计算剩余的四个元素
        for(int j = num_classes % 4 - 1; j >= 0; --j){
            changed[1] += abs(pre_label_function[i][j] - label_function[i][j]);
        }
    }
    while(iter < max_iter && abs(changed[1] - changed[0]) > tol){

/*      if(iter % 10 == 0){
           cout << "Iteration:" << iter << "/" << max_iter << " changed: " << cur_changed << endl;
       }*/
        memcpy(pre_label_function,label_function,sizeof label_function);
        iter += 1;

        float tmp[num_samples][num_classes];
        //SIMD 点乘 tmp[num_samples][num_classes] = aff[num_sample * num_sample] * label_function[num_samples][num_classes]
        __m128 t1, t2, sum;
        //  转置
        float label_function_T[num_classes][num_samples];
        for(int i = 0; i < num_classes; i ++){
            for (int j = 0; j < num_samples; ++j) {
                label_function_T[i][j] = label_function[j][i];
            }
        }
        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                tmp[i][j] = 0.0;
                sum = _mm_setzero_ps();
                for (int k = num_samples - 4; k >= 0; k -= 4) {
                    t1 = _mm_loadu_ps(affinity_matrix[i] + k);
                    t2 = _mm_loadu_ps(label_function_T[j] + k);
                    t1 = _mm_mul_ps(t1, t2);
                    sum = _mm_add_ps(sum, t1);
                }
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                _mm_store_ss(tmp[i] + j, sum);
                // 计算剩余的四个元素
                for (int k = (num_samples % 4) - 1; k >= 0; --k) {
                    tmp[i][j] += affinity_matrix[i][k] * label_function_T[j][k];
                }
            }
        }
        memcpy(label_function,tmp,sizeof tmp);
        //保证原始数据不会跑偏
        memcpy(label_function,clamp_data_label,sizeof clamp_data_label);


        changed[0] = changed[1];
        changed[1] = 0;
        //计算changed，矩阵相减 求绝对值 然后相加
        for(int i = 0; i < num_samples; i ++){
            __m128 t_pre,t_cur;
            __m128 res_change = _mm_setzero_ps();
            for(int j = num_classes - 4; j >= 0; j -=4){
                t_pre = _mm_loadu_ps(pre_label_function[i] + j);
                t_cur = _mm_loadu_ps(label_function[i] + j);
                t_pre = _mm_add_ps(t_pre,t_cur);
                res_change = _mm_add_ps(res_change, _mm_abs_epi16(t_pre));
            }
            res_change = _mm_hadd_ps(res_change,res_change);
            res_change = _mm_hadd_ps(res_change,res_change);
            _mm_store_ss(changed + 1, res_change);

            // 计算剩余的四个元素
            for(int j = num_classes % 4 - 1; j >= 0; --j){
                changed[1] += abs(pre_label_function[i][j] - label_function[i][j]);
            }
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
                1000,
                0.005);
        auto le = clock();
        cout << "数据量为i:"<< num << ";运行时长:" << (double )(le - ls) / CLOCKS_PER_SEC << endl;
        string labels_csv = "test_data_res_" + to_string(num) + ".csv";
        write_test_data(labels_csv,unlabel_data_labels,num);
    }
    return 0;
}

