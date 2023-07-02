clear;
CCdata_DPP = importdata('Simple_back_projection.txt');
CCdata_DPP = CCdata_DPP';
CCdata_DPP = reshape(CCdata_DPP,100,100,100);
CCdata_DPP = permute(CCdata_DPP,[2 1 3]);

figure;
% G=fspecial('gaussian',[3 3], 1);
% cut_slice = imfilter((CCdata_DPP(:,:,50)),G,'same');
cut_slice = CCdata_DPP(:,:,50);
% cut_slice = medfilt2(cut_slice,[3,3]);
imagesc(cut_slice);
colormap(hot(128));
colorbar;

figure,
[x,y,z]=meshgrid(1:100,1:100,1:100);
xslice=[1:5:100];
yslice=[1:5:100];
zslice=[1:5:100];
slice(x,y,z,CCdata_DPP,xslice,yslice,zslice);
shading interp;
colormap(hot);
grid off

xlabel('X'); 
ylabel('Y');
zlabel('Z');
alpha(0.04);
colorbar;
axis([1 100 1 100 1 100]);
% 
a_1 = cut_slice(50,:);
norm_img = mapminmax(a_1, 0, 1);
figure,
plot( norm_img);
len = length(norm_img);
x_1 = [1:len];