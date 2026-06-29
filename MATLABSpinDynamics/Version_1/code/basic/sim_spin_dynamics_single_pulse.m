% Use Martin/Yi-Qiao's terminology to simulate dynamics of spin-1/2
% ensembles due to a single RF pulse.
% ------------------------------------
% tp = RF pulse segment durations in sec
% phi = RF pulse segment phases in radians
% amp = pulse segment amplitude (zero for free precession)
% all coherence pathways are considered in this simulation
% ------------------------------------
% Soumyajit Mandal
% Initial version: 02/19/13

function [Mx,My,Mz]=sim_spin_dynamics_single_pulse(tp,phi,amp,del_w)

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

% Magnetization created by excitation pulse
Mx=zeros(1,numpts); My=Mx; Mz=Mx;

for k=1:numpts
    Mx(k)=(mvect(cp,1,k)+mvect(cm,1,k))/2; % Mx
    My(k)=(mvect(cp,1,k)-mvect(cm,1,k))/(2*i); % My
    Mz(k)=mvect(c0,1,k); % Mz
end

% Plot magnetization created by RF pulse
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