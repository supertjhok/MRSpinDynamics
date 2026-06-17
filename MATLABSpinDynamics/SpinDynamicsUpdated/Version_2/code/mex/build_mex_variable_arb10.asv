% MEX compilation script for sim_spin_dynamics_arb10

% Example input structure (params)
% tp: [1×32 double]
% pul: [2 0 0 3 0 0 4 0 0 5 0 0 6 0 0 7 0 0 8 0 0 9 0 0 10 0 0 11 0 0 12 0]
% amp: [1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0]
% acq: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
% grad: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
% Rtot: {1×12 cell}
% del_w: [1×5000 double]
% del_wg: [1×5000 double]
% w_1: [1×5000 double]
% T1n: [1×5000 double]
% T2n: [1×5000 double]
% m0: [1×5000 double]
% mth: [1×5000 double]

npts=5e3;
% Rotation matrix
mat.R_00=complex(zeros(1,npts)); mat.R_0p=complex(zeros(1,npts)); mat.R_0m=complex(zeros(1,npts));
mat.R_p0=complex(zeros(1,npts)); mat.R_m0=complex(zeros(1,npts)); mat.R_pp=complex(zeros(1,npts));
mat.R_mm=complex(zeros(1,npts)); mat.R_pm=complex(zeros(1,npts)); mat.R_mp=complex(zeros(1,npts));

params.tp=pi*ones(1,32);
params.pul=zeros(1,32);
params.amp=zeros(1,32);
params.acq=zeros(1,32);
params.grad=zeros(1,32);
params.Rtot=cell(1,12);
for i=1:12
    params.Rtot{i}=mat;
end
params.del_w=zeros(1,npts);
params.del_wg=zeros(1,npts);
params.w_1=zeros(1,npts);
params.T1n=zeros(1,npts);
params.T2n=zeros(1,npts);
params.m0=zeros(1,npts);
params.mth=zeros(1,npts);

% Use coder.typeof to specify variable-size inputs
eg.tp=coder.typeof(params.tp,[1 1e4],1);
eg.pul=coder.typeof(params.pul,[1 1e4],1);
eg.amp=coder.typeof(params.amp,[1 1e4],1);
eg.acq=coder.typeof(params.acq,[1 1e4],1);
eg.grad=coder.typeof(params.grad,[1 1e4],1);
eg.Rtot=coder.typeof(params.Rtot,[1,1e4],1);
eg.del_w=coder.typeof(params.del_w,[1 1e6],1);
eg.del_wg=coder.typeof(params.del_wg,[1 1e6],1);
eg.w_1=coder.typeof(params.w_1,[1 1e6],1);
eg.T1n=coder.typeof(params.T1n,[1 1e6],1);
eg.T2n=coder.typeof(params.T2n,[1 1e6],1);
eg.m0=coder.typeof(params.m0,[1 1e6],1);
eg.mth=coder.typeof(params.mth,[1 1e6],1);

% Generate code using coder.typeof to specify
% upper bounds for the example inputs
codegen -report sim_spin_dynamics_arb10.m -args {eg}