import numpy as np
import matplotlib.pyplot as plt


def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):

    color = ['ob', 'og', 'or', 'oc', 'om', 'oy', 'ok', 'ow']
    for i in range(Mat_Unlabel.shape[0]):
        plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], color[int(unlabel_data_labels[i])])

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(0.0, 12.)
    plt.ylim(0.0, 18.)
    plt.title("sacle:" + str(unlabel_data_labels.shape[0]))
    plt.savefig(str(unlabel_data_labels.shape[0]) + '.png')
    plt.show()




def loadBand(num_unlabel_samples):
    array = []
    for i in range(2,18,2):
        array.append([5.0,float(i)])

    Mat_Label = np.asarray(array)
    labels = [i for i in range(8)]  
    num_dim = Mat_Label.shape[1]  
    Mat_Unlabel = np.zeros((num_unlabel_samples, num_dim), np.float32) 
    cnt = num_unlabel_samples // 8
    for i in range(8):
        Mat_Unlabel[i * cnt : (i + 1) * cnt,:] = (np.random.rand(num_unlabel_samples//8, num_dim)) * np.array([1, 1]) + Mat_Label[i]
    np.savetxt("Mat_Label_" + str(num_unlabel_samples) +".csv",Mat_Label,delimiter = ',')
    np.savetxt("Mat_Unlabel_" + str(num_unlabel_samples) + ".csv",Mat_Unlabel,delimiter = ',')
    np.savetxt("labels.csv",labels,delimiter = ',')
    return Mat_Label, labels, Mat_Unlabel  

    

for num in [128,256,512,1024,2048,3072,4096,5120,6144,7168,8192]:
    path = "your result path"
    labels = [0,1]
    Mat_Label = np.loadtxt(path + "/Mat_Label_"+str(num)+".csv",delimiter=',')
    Mat_Unlabel = np.loadtxt(path + "/Mat_Unlabel_"+str(num)+".csv",delimiter=',')
    unlabel_data_labels = np.loadtxt(path + "/test_data_res_"+str(num)+".csv",delimiter=',')
    show(Mat_Label,labels,Mat_Unlabel,unlabel_data_labels)
