close all;
clear all;
clc;
jj=3;
if jj==1
    filename='C:/Users/111/Desktop/图像处理作业/图像去噪/boat.bmp';
end
if jj==2
    filename='C:/Users/111/Desktop/图像处理作业/图像去噪/barb.bmp';
end
if jj==3
    filename='C:/Users/111/Desktop/图像处理作业/图像去噪/lena.bmp';
end
if jj==4
    filename='C:/Users/111/Desktop/图像处理作业/图像去噪/mandrill.bmp';
end
if jj==5
    filename='C:/Users/111/Desktop/图像处理作业/图像去噪/peppers-bw.bmp';
end

    I=imread(filename);
    h = fspecial('gaussian',19, 2 );
    % 构造高斯低通滤波器
    I_blur=imfilter(I,h,'conv');
    J=imnoise(I_blur,'gaussian',0,0.05); % 添加高斯噪声
    J2=imnoise(I_blur,'gaussian',0,0.1); % 添加高斯噪声
    J3=imnoise(I_blur,'gaussian',0,1); % 添加高斯噪声
    figure;
    subplot(141),imshow(I);
    title('原始图像');
    subplot(142),imshow(J,[]); % 显示退化图像(采用高斯低通滤波器对图像进行退化,并添加均值为0,方差为0.05的高斯噪声进一步对图像进行退化)
    title('sigma=0.05');
    subplot(143),imshow(J2,[]); % 显示退化图像(采用高斯低通滤波器对图像进行退化,并添加均值为0,方差为0.1的高斯噪声进一步对图像进行退化)
    title('sigma=0.10');
    subplot(144),imshow(J3,[]); % 显示退化图像(采用高斯低通滤波器对图像进行退化,并添加均值为0,方差为1的高斯噪声进一步对图像进行退化)
    title('sigma=1');

