% Create typical w0 and w1 fields for a single-sided sensor
% Static gradient: y (assumed uniform) (T/m)
% FOV: (x,z) plane (m), uniform field

function [sp]=create_fields_single_sided(sp)

% Fixed offset: strong gradient in y-direction
ny=sp.ny; maxoffs=sp.maxoffs; % Size of y-axis
del_w0y=linspace(-maxoffs,maxoffs,ny); % Linear y-gradient

if sp.plt_fields
    figure;
    plot(del_w0y); title('\Delta\omega_{0} along y')
end

% Pulsed offset: gradients in (x,z) plane
nx=sp.nx; nz=sp.nz; % Size of (x,z) plane
x1=linspace(1,nx,nx); z1=linspace(1,nz,nz);
del_wgx=linspace(-1,1,nx); % Normalized to +/- 1
del_wgz=linspace(-1,1,nz);

del_wgx=repmat(del_wgx',1,nz);
del_wgz=repmat(del_wgz,nx,1);
% Reshape to vectors
del_wgxv=reshape(del_wgx,1,nx*nz);
del_wgzv=reshape(del_wgz,1,nx*nz);

if sp.plt_fields
    figure;
    subplot(1,2,1); imagesc(z1,x1,del_wgx); colorbar; title('G_x');
    subplot(1,2,2); imagesc(z1,x1,del_wgz); colorbar; title('G_z');
end

% Symmetric Gaussian w1 in (x,z) plane, uniform in y
sigma_x=1*nx; sigma_z=1*nz; % Width of gaussian (adjust as needed)
[Z1,X1]=meshgrid(z1,x1); XZ=[X1(:) Z1(:)];
xz=mvnpdf(XZ,[round(nx/2),round(nz/2)],[sigma_x^2,0;0,sigma_z^2]);
xz=reshape(xz,nx,nz);
xz=xz/max(max(xz)); % Normalize to value at center
xzv=reshape(xz,1,nx*nz); % Reshape to vector

if sp.plt_fields
    figure;
    if nx>1 && nz>1
        surf(z1,x1,xz); title('Normalized RF amplitude in (x,z) plane');
    end
end

% Sample properties
rho=sp.rho; T1map=sp.T1map; T2map=sp.T2map;
rhov=reshape(rho,1,nx*nz); % Reshape to vector
T1mapv=reshape(T1map,1,nx*nz);
T2mapv=reshape(T2map,1,nx*nz);

if sp.plt_fields
    figure;
    subplot(1,3,1); imagesc(z1,x1,rho); title('Spin density')
    subplot(1,3,2); imagesc(z1,x1,T1map); title('T1')
    subplot(1,3,3); imagesc(z1,x1,T2map); title('T2')
end

% Create offset frequency, RF amplitude, and sample vectors
% Created by flattening (x,z) for each y to a 1D list, then repeating for each y
sp.numpts=ny*nx*nz;
del_w0=zeros(1,sp.numpts); w_1=del_w0; del_wx=del_w0; del_wz=del_w0;
m0=del_w0; mth=del_w0; T1=del_w0; T2=del_w0;
for i=1:sp.ny
    del_w0((i-1)*nx*nz+1:i*nx*nz)=del_w0y(i)*ones(1,nx*nz);
    del_wx((i-1)*nx*nz+1:i*nx*nz)=del_wgxv;
    del_wz((i-1)*nx*nz+1:i*nx*nz)=del_wgzv;
    w_1((i-1)*nx*nz+1:i*nx*nz)=xzv;
    m0((i-1)*nx*nz+1:i*nx*nz)=rhov;
    mth((i-1)*nx*nz+1:i*nx*nz)=rhov;
    T1((i-1)*nx*nz+1:i*nx*nz)=T1mapv;
    T2((i-1)*nx*nz+1:i*nx*nz)=T2mapv;
end

sp.del_w=del_w0;
sp.w_1=w_1;
sp.del_wx=del_wx;
sp.del_wz=del_wz;
sp.m0=m0; sp.mth=mth; sp.T1=T1; sp.T2=T2;
