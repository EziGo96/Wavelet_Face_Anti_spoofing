
clear
% read image convert to grayscale
X = imread('../../input/Lenna_Orig.png');
X = double(rgb2gray(X));

% example for haar

% matlab implementation
wavelet_name = 'haar';
[LoD,HiD] = wfilters(wavelet_name,'d')
[cA,cH,cV,cD] = dwt2(X,LoD,HiD,'mode','symh');
X_rec = idwt2(cA,cH,cV,cD,wavelet_name);

% Multi-Level Wavelet Convolutional
% Neural Networks implementation

x1 = X(1:2:end,1:2:end)/(2);
x2 = X(2:2:end,1:2:end)/(2);
x3 = X(1:2:end,2:2:end)/(2);
x4 = X(2:2:end,2:2:end)/(2);

x_LL = x1 + x2 + x3 + x4;
x_LH = -x1 - x3 + x2 + x4;
x_HL = -x1 + x3 - x2 + x4;
x_HH = x1 - x3 - x2 + x4;
    

% implementation from theory decomposition
% first downsample image couloms (apply vertical transform) 1d haar
% transform on each column

a_j = ( X(:,1:2:end) + X(:,2:2:end) ) * (1/sqrt(2));
d_j = ( X(:,1:2:end) - X(:,2:2:end) ) * (1/sqrt(2));

% second apply horizontal transform on the rows
ajj = (a_j(1:2:end,:) + a_j(2:2:end,:)) * (1/sqrt(2)); % LL
djh = (a_j(1:2:end,:) - a_j(2:2:end,:)) * -(1/sqrt(2)); % LH
djv = (d_j(1:2:end,:) + d_j(2:2:end,:)) * -(1/sqrt(2)); % HL
djd = (d_j(1:2:end,:) - d_j(2:2:end,:)) * (1/sqrt(2)); % HH

% 3 level decomposition
[LL, LH, HL, HH] = dwt2d_haar(X);
img_rec = idwt2d_haar(LL, LH, HL, HH);

err_img = max(max(X_rec - img_rec));

[LL2,LH2,HL2,HH2] = dwt2d_haar(LL);
[LL3,LH3,HL3,HH3] = dwt2d_haar(LL2);

% errors for Multi-Level Wavelet Convolutional Neural Networks
% err_LL = max(max(cA - x_LL));
% err_LH = max(max(cH - x_LH));
% err_HL = max(max(cV - x_HL));
% err_HH = max(max(cD - x_HH));

% errors for my implementatoin
% err_ajj = max(max(cA - ajj));
% err_djh = max(max(cH - djh));
% err_djv = max(max(cV - djv));
% err_djd = max(max(cD - djd));


err_multi_my_ll = max(max(x_LL-ajj));
err_multi_my_lh = max(max(x_LH-djh));
err_multi_my_hl = max(max(x_HL-djv));
err_multi_my_hh = max(max(x_HH-djd));


downsample_x = X(1:2:end,1:2:end);
figure(1)
% imshow(uint8(X_rec))
histogram(uint8(downsample_x))

figure(2)
imshow(uint8(cD/2))

figure(3)
imshow(uint8(downsample_x))


figure(4)
imshow(uint8(X))

figure(5)
histogram(uint8(cD/2))
