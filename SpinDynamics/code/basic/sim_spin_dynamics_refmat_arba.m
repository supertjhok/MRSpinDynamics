% Use Martin/Yi-Qiao's terminology to simulate dynamics of spin-1/2
% ensembles with precalculated refocusing matrix, refocusing and excitation
% pulses have arbitrary power levels
% ------------------------------------
% tp = durations in us
% phi = phases in radians
% amp = normalized amplitudes = 0 for free precession
% len_acq = acquisition length (us)
% all coherence pathways are considered in this simulation
% ------------------------------------
% Soumyajit Mandal 08/26/10
% Allowed arbitrary pulse amplitudes 02/25/11
% Use normalized w1 03/16/11

function [echo,tvect]=sim_spin_dynamics_refmat_arba(tp,phi,amp,refmat,del_w,len_acq)

i=sqrt(-1);

% Initial magnetization vector amplitude 
m0=1; 

dt=1e-2; % sampling period for observing echo
necho=round(len_acq/dt); % number of time points for observing echo

delt=dt*necho; % Final time period for observing echo
tvect=linspace(-delt/2,delt/2,necho);

% Calculate echo spectrum (no relaxation or diffusion)
numpts=length(del_w);
mvect=zeros(3,1,numpts); % Magnetization vectors
mvect(1,1,:)=m0*ones(1,1,numpts); % Initial mag vectors are along z-axis

% Excitation
num_pulses=length(phi);
for j=1:num_pulses
    if amp(j)>0
        w1=amp(j);
        Omega=sqrt(w1.^2+del_w.^2);
        
        mat=calc_matrix_elements(del_w,w1,Omega,tp(j),phi(j)); % RF pulses
    else
        mat=calc_fp_matrix_elements(del_w,tp(j)); % Free precession
    end
    for k=1:numpts
        mvect(:,:,k)=mat(:,:,k)*mvect(:,:,k);
    end
end

% Only -1 coherence is visible
cm=map_pi(-1);
trans=zeros(1,numpts);

% Refocusing
for k=1:numpts
    mvect(:,:,k)=refmat(:,:,k)*mvect(:,:,k);
    trans(k)=mvect(cm,1,k); % -1 coherence
end

% window function for acquisition only between the 180 pulses
window = sinc(del_w*len_acq/(2*pi));
fy = conv(trans,window);
my = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);

%figure(4);
%plot(del_w,abs(trans)); hold on;
%plot(del_w,abs(my),'r-');

% Calculate time-domain waveform
echo=zeros(1,necho);

for j=1:necho
    echo(j)=sum(my.*exp(-i*del_w*tvect(j))); % -1 coherence at time of detection
end

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

% For free precession, w1 = 0
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