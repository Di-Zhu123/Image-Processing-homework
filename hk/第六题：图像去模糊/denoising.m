%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Copyright (C) 2020 Zaiwen Wen, Haoyang Liu, Jiang Hu
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 实例：Tikhonov 正则化模型用于图片去噪
%
% 对于真实图片 $x\in\mathcal{R}^{m\times n}$ 和带噪声的图片 $y=x+e$（其中 $e$ 是高斯白噪声）。
% Tikhonov 正则化模型为：
%
% $$ \displaystyle \min_xf(x)=\frac{1}{2}\|x-y\|^2_F
% +\lambda(\|D_1 x\|_F^2+\|D_2x\|_F^2), $$
%
% 其中 $D_1x$, $D_2x$ 分别表示 $x$ 在水平和竖直方向上的向前差分， $\lambda$ 为正则化系数。
% 上述优化问题的目标函数中，第二项要求恢复的 $x$ 有较好的光滑性，以达到去噪的目的。
% 注意到上述目标函数是可微的，我们利用结合BB步长和非精确搜索的
% 的梯度下降对其进行求解。
%
%% 图片和参数准备
% 设定随机种子。
% clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%%
% 载入未加噪的原图作为参考，记录为 |u0| 。
% u = load ('tower.mat');
% u = u.B1;
% u = double(u);
u=double(I);
[m,n] = size(u);
u0 = u;
%%%
% 生成加噪的图片，噪声 $e$的每个元素服从独立的高斯分布 $\mathcal{N}(0,20^2)$
% ，并对每个像素进行归一化处理（将像素值转化到[0,1]区间内）。注意到 MATLAB 的 |imshow|
% 函数（当第二个参数设定为空矩阵时），能够自动将矩阵中最小的元素对应到黑色，将最大的元素对应为白色。
% u = u + 20*randn(m,n);
% maxu = max(u(:)); minu = min(u(:));
% u = (u - minu)/(maxu - minu);
%%%
u=double(J);
% 参数设定，以一个结构体提供各参数，分别表示 $x$，梯度和函数值的停机标准，输出的详细程度，和最大迭代次数。
opts = struct();
opts.xtol = 1e-8;
opts.gtol = 1e-6;
opts.ftol = 1e-16;
opts.record  = 0;
opts.maxit = 200;
%% 求解正则化优化问题
% 分别取正则化系数为 $\lambda=0.5$ 和 $\lambda=2$ ，利用带BB 步长的梯度下降求解对应的优化问题，见<fminGBB.html 带BB步长线搜索的梯度法> 。
lambda = 0.5;
fun = @(x) TV(x,u,lambda);
[x1,~,out1] = fminGBB(u,fun,opts);
re1=PSNR(u0,x1)
lambda = 2;
fun = @(x) TV(x,u,lambda);
[x2,~,out2] = fminGBB(u,fun,opts);
re2=PSNR(u0,x2)
re3=PSNR(u0,double(wnr1))
lam_arr=[0:0.2:3];
re=[0:0.2:3];
i=1;
% for lam=[0:0.2:3]
%     fun = @(x) TV(x,u,lam);
%     [xn,~,out1] = fminGBB(u,fun,opts);
%     re(1,i)=PSNR(u0,xn);
%     i=i+1;
% end

% plot(lam_arr',re');
hold on;
%%%
% 结果可视化，将不同正则化系数的去噪结果以图片形式展示。
figure;
subplot(1,5,1);
imshow(u0,[]);
title('原图')
subplot(1,5,2);
imshow(u,[]);
title('退化图像')
subplot(1,5,3);
imshow(x1,[]);
title('MAP/Sobolev正则化')
subplot(1,5,4);
imshow(x2,[]);
title('MAP/TV正则化')
subplot(1,5,5);
imshow(double(wnr1),[]);
title('MLE')

%% Tikhonov 正则化模型的目标函数值和梯度计算
% 该无约束优化问题的目标函数为：
%
% $$f(x) = \frac{1}{2}\|x-y\|_F^2 + \lambda(\|D_1x\|_F^2+\|D_2x\|_F^2). $$
%
function [f,g] = TV(x,y,lambda)
%%%
% $y, \lambda$ 分别表示带噪声图片和正则化参数， |f| ， |g| 表示在 |x| 点处的目标函数值和梯度。 
%
% 第一项 $\frac{1}{2}\|x-y\|_F^2$ 用于控制去噪后的图片 $x$和带噪声的图片 $y$之间的距离。
f = .5*norm(x - y, 'fro')^2;
%%%
% 计算两个方向上的离散差分， $(D_1x)_{i,j}=x_{i+1,j}-x_{i,j}$,
% $(D_2x)_{i,j}=x_{i,j+1}-x_{i,j}$。
[m,n] = size(y);
dx = zeros(m,n); dy = zeros(m,n); d2x = zeros(m,n);
for i = 1:m
    for j = 1:n
        ip1 = min(i+1,m); jp1 = min(j+1,n);
        im1 = max(i-1,1); jm1 = max(j-1,1);
        dx(i,j) = x(ip1,j) - x(i,j);
        dy(i,j) = x(i,jp1) - x(i,j);
        %%%
        % 离散的拉普拉斯算子 |d2x| : $(\Delta
        % x)_{i,j}=x_{i+1,j}+x_{i,j+1}+x_{i-1,j}+x_{i,j-1}-4x_{i,j}$。
        d2x(i,j) = x(ip1,j) + x(im1,j) + x(i,jp1) + x(i,jm1) - 4*x(i,j);
    end
end
%%%
% 计算目标函数的第二项（Tikhonov 正则化）并与第一项合并得到当前点处的目标函数值。
f = f + lambda * (norm(dx,'fro')^2 + norm(dy,'fro')^2);
%%%
% 目标函数的梯度可以解析地写出：
%
% $$\nabla f(x)=x-y-2\lambda \Delta x.$$
%
g = x - y - 2*lambda*d2x;
end
function [ output ] = PSNR( img1,img2)
%PSNR 峰值信噪比
    if sum(sum(img1-img2)) == 0
        error('Those pictures are the same');
    end
    MAX=1;          %图像有多少灰度级（我这里定为1）
    % 归一化
    if (max(max(img1))-min(min(img1))) ~= 0
        img1 = (img1-min(min(img1)))./(max(max(img1))-min(min(img1)));
    end
    if (max(max(img1))-min(min(img1))) ~= 0
        img2 = (img2-min(min(img2)))./(max(max(img2))-min(min(img2)));
    end
    %
    MSE=sum(sum((img1-img2).^2))/(1024*1024);     %图片像素设为1024 x 1024
    output=20*log10(MAX/sqrt(MSE));           %峰值信噪比    
end
