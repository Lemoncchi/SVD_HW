from numpy import *
import cv2
from PIL import Image
import pywt
from scipy.misc import *

def modify(ori_img,w_img,a):
    ori_img = array(ori_img.convert('L'))   # 转成灰度图
    w_img = array(w_img.convert('L'))   # 转成灰度图
    cA, (cH, cV, cD) = pywt.dwt2(ori_img, 'haar')   #载体图像dwt变换
    cA1 = cA+a*w_img   #嵌入后载体图像的新的低频分量
    img_with_w= pywt.idwt2((cA1, (cH, cV, cD)), 'haar')

    return img_with_w

def extract(ori_img,img_with_w,w_img,a):
    ori_img = array(ori_img.convert('L'))   # 载体图像原图转成灰度图
    w_img = array(w_img.convert('L'))   # 转成灰度图
    cA, (cH, cV, cD) = pywt.dwt2(ori_img, 'haar')   #载体图像dwt变换
    cA_imgw, (cH, cV, cD) = pywt.dwt2(img_with_w, 'haar')   #含水印载体图像dwt变换
    ex_w = (cA_imgw-cA)/a

    return ex_w

if __name__ == '__main__':
    
    #原始图片
    ori_img = Image.open('lena.jpg')
    w_img = Image.open('wm.jpg')
    
#嵌入水印
    #水印程度
    a = 0.1

    for i in range(10):
        img_with_img = modify(ori_img,w_img,a)
        cv2.imwrite(name[i], img_with_img)
        # 读取图片
        img2 = cv2.imread(name[i])
        # 计算PSNR
        psnrnum = psnr(img1, img2)
        psnrs.append(psnrnum)

    img_with_w = modify(ori_img,w_img,a)
    img_with_w = Image.fromarray(uint8(img_with_w))
    img_with_w.save('im&wm.jpg')
    
    #提取水印
    ex_w = extract(ori_img,img_with_w,w_img,a)    
    wm = Image.fromarray(uint8(ex_w),)
    wm.save('ex_wm.jpg')

    # 绘制折线图
    plt.plot(range(1,11), psnrs, marker='o')
    plt.xlabel('Watermark Embedding Strength')
    plt.ylabel('PSNR')
    plt.savefig('chart.jpg')
