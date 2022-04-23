clc, clear
%%%%%%%%%??Fea%%%%%%%%%%%%%
load T.mat
[rows, cols] = size(T);
tic;
Fea = extract_features(rows, cols, T);
toc;
Fea = abs(Fea);
%save Fea.mat Fea
%%%%%%%%%??Fea%%%%%%%%%%%%%
% load('Salinas_corrected.mat', 'salinas_corrected');
% Fea = salinas_corrected; clear salinas_corrected;
%%%%%%%%%%%???????1%%%%%%%%%%
%load Fea.mat
[m, n, d] = size(Fea);
Fea_V = f_tensor2f_vector(Fea);
%Fea_V = Fea_V';
%%%%%%%%%%???????1%%%%%%%%%%
%Fea_V = normc(Fea_V);
%%%%%%%%%%%???????2%%%%%%%%%%
save Fea_V.mat Fea_V
%%%%%%%%%%%???????2%%%%%%%%%%
% 
% Fea_T = f_vector2f_tensor(Fea_V, m);
% save Fea_T.mat Fea_T
% 
% Width = 3; % 近邻窗户大小
% Fea_3D = add_spatial_info(Fea_T, Width);
% save Fea_3D.mat Fea_3D
