%%
% CLAB2 Task-1: Harris Corner Detector
% Your name (Your uniID)
%
sigma = 2; thresh = 0.01; % Parameters, add more if needed
% Derivative masks
dx = [-1 0 1;-1 0 1; -1 0 1];
dy = dx'; % dx is the transpose matrix of dy
% compute x and y derivatives of image
Ix = conv2(bw,dx,'same');
Iy = conv2(bw,dy,'same'); 

g = fspecial('gaussian',max(1,fix(3*sigma)*2+1),sigma);
Ix2 = conv2(Ix.^2,g,'same'); % x and x
Iy2 = conv2(Iy.^2,g,'same'); % y and y
Ixy = conv2(Ix.*Iy,g,'same'); % x and y

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task: Compute the Harris Cornerness                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task: Perform non-maximum suppression and             %
%       thresholding, return the N corner points        %
%       as an Nx2 matrix of x and y coordinates         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

