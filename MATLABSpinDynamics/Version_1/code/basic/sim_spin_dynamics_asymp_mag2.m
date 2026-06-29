% Use Martin/Yi-Qiao's terminology to simulate dynamics of spin-1/2
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

function [masy]=sim_spin_dynamics_asymp_mag2(tp,phi,amp,neff,del_w,t_acq)

i=sqrt(-1);

% Initial magnetization vector amplitude
m0=1;

% Uniform distribution of del_w0 (uniform gradient)
numpts=length(del_w); % number of spin vectors to simulate

% Calculate echo spectrum (no diffusion)
mvect=zeros(3,1,numpts); % Magnetization vectors
mvect(1,1,:)=m0*ones(1,1,numpts); % Initial mag vectors are along z-axis

num_pulses=length(phi);
for j=1:num_pulses
    
    if amp(j)>0
        w1=amp(j);
        Omega=sqrt(w1^2+del_w.^2);
        mat=calc_matrix_elements(del_w,w1,Omega,tp(j),phi(j)); % RF pulses
    else
        mat=calc_fp_matrix_elements(del_w,tp(j)); % Free precession
    end
    
   % for k=1:numpts
   %     mvect(:,:,k)=mat(:,:,k)*mvect(:,:,k);
   % end
    mvect=multiprod(mat,mvect); 
    
end

% Convert magnetization to [x y z] form
c0=map_pi(0); cm=map_pi(-1); cp=map_pi(+1);
tmp=zeros(3,1,numpts);

% Magnetization created by excitation pulse
% Mx=zeros(1,numpts); My=Mx; Mz=Mx;

for k=1:numpts
    tmp(1,1,k)=(mvect(cp,1,k)+mvect(cm,1,k))/2; % Mx
    tmp(2,1,k)=(mvect(cp,1,k)-mvect(cm,1,k))/(2*i); % My
    tmp(3,1,k)=mvect(c0,1,k); % Mz
%    Mx(k)=tmp(1,1,k);
%    My(k)=tmp(2,1,k);
%    Mz(k)=tmp(3,1,k);
end
mvect=tmp;

% Plot magnetization created by excitation pulse
% figure(11);
% plot(del_w,real(Mx),'b-'); hold on;
% plot(del_w,imag(Mx),'b--');
% 
% figure(12);
% plot(del_w,real(My),'r-'); hold on;
% plot(del_w,imag(My),'r--');
% 
% figure(13);
% plot(del_w,real(Mz),'k-'); hold on;
% plot(del_w,imag(Mz),'k--');

trans=zeros(1,numpts);
% Asymptotic magnetization is spin-locked to the effective rotation axis
% Only -1 coherence is visible
% Phase cycling will cancel contributions from z-magnetization
for k=1:numpts
    %trans(k)=neff(2,1,k)*mvect(2,1,k)*neff(2,1,k); % My - with phase cycling
    
    mvect(:,1,k)=dot(neff(:,1,k),mvect(:,1,k)).*neff(:,1,k);
    trans(k)=mvect(1,1,k)-i*mvect(2,1,k); % -1 coherence - without phase cycling
end

% window function for acquisition only between the 180 pulses
window = sinc(del_w*t_acq/(2*pi));
fy = conv(trans,window);
masy = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);

% Mapping from coherence to index
% [0,-1,+1] -> [1,2,3]
function [ind]=map_pi(coh)

if coh==0
    ind=1;
else if coh==-1
        ind=2;
    else
        ind=3;
    end
end

% Calculate matrix elements for RF pulses, neglect relaxation
function R = calc_matrix_elements(del_w,w1,Omega,tp,phi)

i=sqrt(-1);
R_pp=0.5*((w1./Omega).^2+(1+(del_w./Omega).^2).*cos(Omega*tp))+i*(del_w./Omega).*sin(Omega*tp);
R_mm=conj(R_pp);
R_00=(del_w./Omega).^2+(w1./Omega).^2.*cos(Omega*tp);
R_p0=(w1./Omega).*((del_w./Omega).*(1-cos(Omega*tp))-i*sin(Omega*tp))*exp(i*phi);
R_m0=conj(R_p0);
R_0p=0.5*(w1./Omega).*((del_w./Omega).*(1-cos(Omega*tp))-i*sin(Omega*tp))*exp(-i*phi);
R_0m=conj(R_0p);
R_pm=0.5*(w1./Omega).^2.*(1-cos(Omega*tp))*exp(i*2*phi);
R_mp=conj(R_pm);

% Rotation matrix
R=zeros(3,3,length(del_w));
R(1,1,:)=R_00; R(1,2,:)=R_0m; R(1,3,:)=R_0p;
R(2,1,:)=R_m0; R(2,2,:)=R_mm; R(2,3,:)=R_mp;
R(3,1,:)=R_p0; R(3,2,:)=R_pm; R(3,3,:)=R_pp;

% For free precession, w1 = 0, neglecting relaxation
function R = calc_fp_matrix_elements(del_w,tf)

i=sqrt(-1);
R_pp=cos(del_w*tf)+i*sin(del_w*tf);
R_mm=conj(R_pp);
R_00=ones(1,length(del_w));
R_p0=zeros(1,length(del_w));
R_m0=conj(R_p0);
R_0p=zeros(1,length(del_w));
R_0m=conj(R_0p);
R_pm=zeros(1,length(del_w));
R_mp=conj(R_pm);

% Rotation matrix
R=zeros(3,3,length(del_w));
R(1,1,:)=R_00; R(1,2,:)=R_0m; R(1,3,:)=R_0p;
R(2,1,:)=R_m0; R(2,2,:)=R_mm; R(2,3,:)=R_mp;
R(3,1,:)=R_p0; R(3,2,:)=R_pm; R(3,3,:)=R_pp;