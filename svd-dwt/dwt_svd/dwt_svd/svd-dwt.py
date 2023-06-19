import cv2
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

class Watermarking():#水印嵌入提取算法
    def watermark(self, a, ori_img, w_img, level=3):
        self.a = a
        self.img_path = ori_img#.split('/')[-1]
        self.wimg_path = w_img#.split('/')[-1]
        self.level = level

        img = cv2.imread(self.img_path, 0)  # 读取灰度图
            
        Coefficients = pywt.wavedec2(img, wavelet='haar', level=self.level)  # 三级离散小波变换
        self.shape_LL = Coefficients[0].shape  # Coefficients[0] is LL 低频系数

        Uc, Sc, Vc = np.linalg.svd(Coefficients[0])  # 奇异值分解

        W = cv2.imread(self.wimg_path, 0)  # 水印
        self.wshape = W.shape
        if Sc.shape[0] != W.shape[0]:
            W = cv2.resize(W, (Sc.shape[0],Sc.shape[0]))

        SLL = np.zeros(self.shape_LL)  # 奇异值由列表变成矩阵
        row = min(self.shape_LL)
        SLL[:row, :row] = np.diag(Sc)
        self.Sc = SLL  # LL的奇异值

        self.Snew = np.zeros((min(self.shape_LL), min(self.shape_LL)))  # 创建全零矩阵，来接收参数
        for y in range(0, min(self.shape_LL)):
            for x in range(0, min(self.shape_LL)):
                self.Snew[y][x] = self.Sc[y][x] + self.a * (W[y][x])  # 每个像素点依次叠加

        Uw, Sw, Vw = np.linalg.svd(self.Snew)  # 奇异值分解

        LLnew = np.zeros((min(SLL.shape), min(SLL.shape)))  # 创建全零矩阵，来接收参数
        LLnew = Uc.dot(np.diag(Sw)).dot(Vc)
        Coefficients[0] = LLnew
        img_with_img = pywt.waverec2(Coefficients, 'haar')
        return img_with_img

    def extract(self, img_with_img):  #提取

        Cw = pywt.wavedec2(img_with_img, wavelet='haar', level=self.level)  # 小波变换

        Ucw, Scw, Vcw = np.linalg.svd(Cw[0])  # 对低频系数LL进行奇异值分解

        Uw, Sw, Vw = np.linalg.svd(self.Snew)  # 对添加水印后的矩阵进行奇异值分解

        Snew1 = Uw.dot(np.diag(Scw)).dot(Vw)  # 用低频系数LL的奇异值来还原添加水印后的矩阵

        Wdnew = np.zeros((min(self.shape_LL), min(self.shape_LL)))  # 创建全零矩阵，来接收参数

        for y in range(0, min(self.shape_LL)):  # 还原水印
            for x in range(0, min(self.shape_LL)):
                Wdnew[y][x] = (Snew1[y][x] - self.Sc[y][x]) / self.a
        ex_w = cv2.resize(Wdnew,(self.wshape[1],self.wshape[0])) #重新变为原水印大小
        return ex_w
    
if __name__ == '__main__':
    wm = Watermarking()
    ori_img = 'lena.jpg'
    w_img = 'gray_logo.jpg' 
     
    img_with_img = wm.watermark(0.7,ori_img,w_img)
    savename = 'img_with_img.jpg'
    cv2.imwrite(savename, img_with_img)
    print("水印嵌入成功！")
    print("含水印图片为：",savename)

    ex_w=wm.extract(img_with_img)
    savename = 'ex_w2.jpg'
    cv2.imwrite(savename, ex_w)
    print("水印提取成功！")
    print("水印图片为：",savename)
    #name=['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg']
    #img1 = cv2.imread('lena.jpg')  # 读取灰度图
    #psnrs = []
    #for i in range(10):
    #    img_with_img = wm.watermark(i*0.1,ori_img,w_img)
    #    cv2.imwrite(name[i], img_with_img)
    #    # 读取图片
    #    img2 = cv2.imread(name[i])
    #    # 计算PSNR
    #    psnrnum = psnr(img1, img2)
    #    psnrs.append(psnrnum)
    ## 绘制折线图
    #plt.plot(range(1,11), psnrs, marker='o')
    #plt.xlabel('Watermark Embedding Strength')
    #plt.ylabel('PSNR')
    #plt.savefig('chart.jpg')