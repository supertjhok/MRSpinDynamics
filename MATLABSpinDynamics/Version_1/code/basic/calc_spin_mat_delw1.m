% Use Martin/Yi-Qiao's terminology to calculate spin rotation matrix
% Allow RF field to be inhomogeneous
% ------------------------------------
% T_90 = duration of 90 degree RF pulse
% del_w1 = maximum fractional decrease in B1
% tp = RF pulse durations in us
% phi = RF pulse phases in radians
% tf = spacing between pulses in us
% T1, T2 = relaxation times (ms)
% all coherence pathways are considered in this simulation
% all RF pulses are assumed to have the same power level
% ------------------------------------
% Soumyajit Mandal 08/26/10

function [mat]=calc_spin_mat_delw1(T_90,del_w1,tp,phi,tf)

% Units conversions
T_90=T_90/1e6;
tp=tp/1e6;
tf=tf/1e6;

% Uniform distribution of del_w0 (uniform gradient)
maxoffs=10;
numpts=1e4; % number of spin vectors to simulate
cvec=linspace(-maxoffs,maxoffs,numpts);

% B1 distribution
w1_max=pi/(2*T_90); % nominal B1 (assumed to be at resonance)
%w1=w1_max*(1-del_w1*(1-cos(cvec*pi/(2*maxoffs)))); % Smooth distribution
w1=w1_max*(1+del_w1*(rand(1,numpts)-0.5)); 

% B0 distribution
%del_w=w1_max*maxoffs*(2*rand(1,numpts)-1);
del_w=w1_max*cvec;

Omega=sqrt(w1.^2+del_w.^2);

% Calculate net rotation matrix (no relaxation or diffusion)

num_pulses=length(phi);
for j=1:num_pulses
    if j==1
        mat=calc_matrix_elements(del_w,w1,Omega,tp(j),phi(j)); % RF pulses
    else
        tmp=calc_matrix_elements(del_w,w1,Omega,tp(j),phi(j));
        for k=1:numpts
            mat(:,:,k)=tmp(:,:,k)*mat(:,:,k);
        end
    end
    
    if tf(j)>0
        tmp=calc_fp_matrix_elements(del_w,tf(j)); % Free precession
        for k=1:numpts
            mat(:,:,k)=tmp(:,:,k)*mat(:,:,k);
        end
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