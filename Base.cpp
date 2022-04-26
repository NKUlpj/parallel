#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include "time.h"
#include <arm_neon.h>


using namespace std;


typedef unsigned int uint;
const int maxN = 1e4 +  10; // 最大样本数
const int maxF = 10;  // 最大特征数
float affinity_matrix[maxN][maxN];
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
                float module = 0;
                for(int k = features - 4; k >= 0; k -=4){
                    ta = vld1q_f32(MatX[i] + k);
                    tb = vld1q_f32(MatX[j] + k);
                    ta = vsubq_f32(ta, tb);
                    res4 = vaddq_f32(ta,ta);
                }
                float32x2_t suml2 = vget_low_f32(res4);
                float32x2_t sumh2 = vget_high_f32(res4);
                suml2 = vpadd_f32(suml2, sumh2);
                module = (float)vpadds_f32(suml2);
                for(int k = features % 4 - 1; k >= 0; k --){
                    module += (MatX[i][k] - MatX[j][k]) * (MatX[i][k] - MatX[j][k])
                }
                affinity_matrix[i][j] = exp(-(module)/(rbf_sigma * rbf_sigma));
            }
        }
         * */
        cout << "todo" << endl;
        exit(1);
    } else{
        for(int i = 0; i < num_samples; i ++){
            // 计算每个点的邻居，为了画图方便，测试案例使用的是二维数据，此处没有使用SIMD加速
            vector<float> squaredDist(num_samples);
            for(int j = 0; j < num_samples; j ++){
                for(int k = 0; k < 2; k ++){
                    float residual = MatX[j][k] - MatX[i][k];
                    squaredDist[j] += residual * residual;
                }
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
    for(int i = 0; i < num_samples; i ++){
        for(int j = 0; j < num_classes; j ++){
            cur_changed += abs(pre_label_function[i][j] - label_function[i][j]);
        }
    }
    while(iter < max_iter && abs(pre_changed - cur_changed) > tol){

/*      if(iter % 10 == 0){
           cout << "Iteration:" << iter << "/" << max_iter << " changed: " << cur_changed << endl;
       }*/
       memcpy(pre_label_function,label_function,sizeof label_function);
       iter += 1;

        float tmp[num_samples][num_classes];
        for(int i = 0; i < num_samples; i ++){
            for(int j = 0; j < num_classes; j ++){
                tmp[i][j] = 0.00000f;
                for(int k = 0; k < num_samples; k ++){
                    tmp[i][j] += affinity_matrix[i][k] * label_function[k][j];
                }
            }
        }
        memcpy(label_function,tmp,sizeof tmp);
        //保证原始数据不会跑偏
        memcpy(label_function,clamp_data_label,sizeof clamp_data_label);

        pre_changed = cur_changed;
        cur_changed = 0;
       for(int i = 0; i < num_samples; i ++){
            for(int j = 0; j < num_classes; j ++){
                cur_changed += abs(pre_label_function[i][j] - label_function[i][j]);
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

