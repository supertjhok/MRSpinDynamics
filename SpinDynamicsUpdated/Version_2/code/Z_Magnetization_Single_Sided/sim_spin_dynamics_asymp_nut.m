% Use Hurlimann-Song terminology to simulate dynamics of spin-1/2
% ensembles. Find asymptotic echo shape from effective rotation axis.
% ------------------------------------
% tp = RF pulse durations in sec
% phi = RF pulse phases in radians
% amp = pulse amplitude (zero for free precession)
% neff = effective rotation axis
% t_acq = duration for observing echo
% all coherence pathways are considered in this simulation
% ------------------------------------
% Soumyajit Mandal
% Initial version: 08/26/10
% Allow arbitrary pulse amplitudes: 03/03/11
% Use multiprod for 3D matrix multiplication: 09/13/11 (significant
% speedup!)
% Use completely vectorized code for further speed (~3x speedup): 02/25/13

function [mz]=sim_spin_dynamics_asymp_nut(pulseLength,phi,amp,neff,del_w,t_acq)

tp = pulseLength;


m0=1; % Initial magnetization vector amplitude
numpts=length(del_w); % number of spin vectors to simulate
window = sinc(del_w*t_acq/(2*pi)); % window function for acquisition
window = window./sum(window);

% Calculate magnetization spectrum (no diffusion)
mvect=zeros(3,numpts); % Magnetization vectors
mvect(1,:)=m0*ones(1,numpts); % Initial vectors are along z-axis

num_pulses=length(phi);
for j=1:num_pulses
    %Calculate z magnetization
    w1=amp(j);
    Omega=sqrt(w1^2+del_w.^2);
    mat=calc_matrix_elements(del_w,w1,Omega,tp(j),phi(j)); % RF pulses  
    
    tmp=mvect;
    mvect(1,:)=mat.R_00.*tmp(1,:)+mat.R_0m.*tmp(2,:)+mat.R_0p.*tmp(3,:); % M0
    mvect(2,:)=mat.R_m0.*tmp(1,:)+mat.R_mm.*tmp(2,:)+mat.R_mp.*tmp(3,:); % M-
    mvect(3,:)=mat.R_p0.*tmp(1,:)+mat.R_pm.*tmp(2,:)+mat.R_pp.*tmp(3,:); % Mz   
end

% Magnetization created by excitation pulse
% Convert magnetization to [x y z] form
tmp=mvect;
mvect(1,:)=0.5*(tmp(3,:)+tmp(2,:)); % Mx
mvect(2,:)=-0.5*1i*(tmp(3,:)-tmp(2,:)); % My
mvect(3,:)=tmp(1,:); % Mz
% plot(del_w,fftshift(abs(fft(mvect(1,:)))));
mz = mvect(3,:);

% Calculate asymptotic magnetization (spin-locked to effective rotation
% axis)
% Only -1 coherence is visible
% Phase cycling will cancel contributions from z-magnetization
% trans=dot(mvect,neff).*(neff(1,:)-1i*neff(2,:)); % (Mx - iMy) = M-
% masy = conv(trans,window,'same');

% Calculate matrix elements for RF pulses, neglect relaxation
function R = calc_matrix_elements(del_w,w1,Omega,tp,phi)

dw=del_w./Omega; w1n=w1./Omega; ph=exp(1i*phi);
sn=sin(Omega*tp); cs=cos(Omega*tp);

R.R_00=dw.^2+w1n.^2.*cs;
R.R_0p=0.5*w1n.*(dw.*(1-cs)-1i*sn)/ph; R.R_0m=conj(R.R_0p);
R.R_p0=w1n.*(dw.*(1-cs)-1i*sn)*ph; R.R_m0=conj(R.R_p0);
R.R_pp=0.5*(w1n.^2+(1+dw.^2).*cs)+1i*dw.*sn; R.R_mm=conj(R.R_pp);
R.R_pm=0.5*w1n.^2.*(1-cs)*ph.^2; R.R_mp=conj(R.R_pm);

% For free precession, w1 = 0, neglecting relaxation
function R = calc_fp_matrix_elements(del_w,tf)

numpts=length(del_w);
R.R_00=ones(1,numpts);
R.R_0p=zeros(1,numpts); R.R_0m=conj(R.R_0p);
R.R_p0=zeros(1,numpts); R.R_m0=conj(R.R_p0);
R.R_pp=cos(del_w*tf)+1i*sin(del_w*tf); R.R_mm=conj(R.R_pp);
R.R_pm=zeros(1,numpts); R.R_mp=conj(R.R_pm);
