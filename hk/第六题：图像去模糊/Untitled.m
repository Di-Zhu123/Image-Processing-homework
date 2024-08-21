close all;
clear all;
clc;
jj=5;
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

    figure;
    subplot(141),imshow(I);
    title('原始图像');
    subplot(142),imshow(J,[]); % 显示退化图像(采用高斯低通滤波器对图像进行退化,并添加均值为0,方差为0.001的高斯噪声进一步对图像进行退化)
    title('退化图像');

    h1 = fspecial('gaussian', 512, 300 );
    HC=1./h1;
    K=fftfilter(J,HC);% 逆滤波操作
    
    imwrite(K,'C:/Users/111/Desktop/图像处理作业/图像去噪/inv/barb.bmp');
    subplot(143),imshow(K,[]);% 逆滤波复原,频率大
    title('逆滤波时得到的图像');

    wnr1 = wiener2(J);
    wnr2=deconvwnr(J,h,0);
    subplot(144),imshow(wnr1,[]);
    title('Wiener');






function Z = fftfilter(X,H)
% 图像的频域滤波处理
% X为输入图像
% H为滤波器
% Z为输出图像
F=fft2(X,size(H,1),size(H,2));% 傅里叶变换
Z=H.*F;                       % 频域滤波
Z=abs(ifft2(ifftshift(Z)));   % 傅里叶反变换
Z=Z(1:size(X,1),1:size(X,2));
end
