import os
from PIL import Image
from PIL import ImageFilter
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from skimage import data,exposure
from skimage.transform import pyramid_gaussian,pyramid_expand


def img_process_guassion(image, file):
    
    image=np.asarray(image)
    rows=cols=512
    # pyramid = tuple(pyramid_gaussian(image, downscale=2,mode=114514))
    pyramid = tuple(pyramid_gaussian(image, downscale=2,sigma=5))
    # determine the total number of rows and columns for the composite
    composite_rows = max(rows, sum(p.shape[0] for p in pyramid[1:]))
    composite_cols = cols + pyramid[1].shape[1]
    composite_image = np.zeros((composite_rows, composite_cols),
                            dtype=np.double)

    # store the original to the left
    composite_image[:rows, :cols] = pyramid[0]

    # stack all downsampled images in a column to the right of the original
    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    # fig, ax = plt.subplots()
    # ax.imshow(composite_image, cmap='gray')
    # plt.show()
    residual=[None]*(len(pyramid))
    for i in range(len(pyramid)-1):
        residual[i]=pyramid[i]-pyramid_expand(pyramid[i+1],order=3, mode='wrap')
    residual[-1]=pyramid[-1]
    composite_image = np.zeros((composite_rows, composite_cols),
                            dtype=np.double)
    composite_image[:rows, :cols] = residual[0]
    i_row = 0
    for p in residual[1:]:
        n_rows, n_cols = p.shape[:2]

        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    # fig, ax = plt.subplots()
    # ax.imshow(composite_image, cmap='gray')
    # plt.show()

    return pyramid,residual


def img_process(image,  file):
    pass

# 指定文件夹路径
folder_path = r'C:/Users/111/Desktop/图像处理作业/图像去噪'
residuals=[]
pyramids=[]
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    
    # 检查文件是否为图片
    if filename.endswith('.bmp'):
        # print(1)
        # 构建图片的完整路径
        image_path = os.path.join(folder_path, filename)

        # 使用PIL库打开图片
        image = Image.open(image_path).convert('L')

        # 对图片进行处理
        file = filename[:-4]
        pyramid,residual=img_process_guassion(image,  file)
        residuals.append(residual)
        pyramids.append(pyramid)
        # 关闭图片
        image.close()
fig, axes = plt.subplots(2, 5, figsize=(9, 9))
bin_num=32
print(len(pyramids))
for j in range(len(pyramids[0])):
    images=[]
    hist_sum=[0]*bin_num
    if j>=5:
        break
    for i, img in enumerate(pyramids):
        img=np.asarray(img[j])
        images.append(img.astype(np.float32) )
    
    hist = cv2.calcHist(images,[0],None,[bin_num],[0,1],accumulate=True)
    hist=hist/5/512/512
    # print(hist)
    bins=np.linspace(0,1,bin_num)
    axes[0,j].plot(bins, hist,label=f'j={j}')
    axes[0,j].set_title(f'{j}th pyramid')
    axes[0,j].set_xlabel('Pixel Value')
    axes[0,j].set_ylabel('Frequency')

    images2=[]
    for i, img in enumerate(residuals):
        img=np.asarray(img[j])
        images2.append(img.astype(np.float32))
    hist2 = cv2.calcHist(images2,[0],None,[bin_num],[0,1],accumulate=True)
    hist2=hist2/5/512/512
    axes[1,j].plot(bins, hist2,label=f'j={j}')
    axes[1,j].set_title(f'{j}th residual')
    axes[1,j].set_xlabel('Pixel Value')
    axes[1,j].set_ylabel('Frequency')


plt.show()