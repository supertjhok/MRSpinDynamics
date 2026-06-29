% Use Martin/Yi-Qiao's terminology to calculate spin rotation matrix
% Pulses can have arbitrary amplitudes
% ------------------------------------
% tp = durations in sec
% phi = phases in radians
% amp = normalized amplitude (zero = free precession)
% tf = spacing between pulses in sec
% all coherence pathways are considered in this simulation
% ------------------------------------
% Soumyajit Mandal
% Original version 08/26/10
% Allowed arbitrary pulse amplitudes 02/25/11

function [mat]=calc_spin_mat_arba_delw(tp,phi,amp,del_w)

numpts=length(del_w); % number of spin vectors to simulate

% Calculate net rotation matrix (no relaxation or diffusion)
if amp(1)>0
    w1=amp(1);
    Omega=sqrt(w1.^2+del_w.^2);
    mat=calc_matrix_elements(del_w,w1,Omega,tp(1),phi(1));  % RF pulses
else
    mat=calc_fp_matrix_elements(del_w,tp(1)); % Free precession
end

num_pulses=length(phi);
for j=2:num_pulses
    if amp(j)>0
        w1=amp(j);
        Omega=sqrt(w1.^2+del_w.^2);
        tmp=calc_matrix_elements(del_w,w1,Omega,tp(j),phi(j)); % RF pulses
    else
        tmp=calc_fp_matrix_elements(del_w,tp(j)); % Free precession
    end
    
    for k=1:numpts
        mat(:,:,k)=tmp(:,:,k)*mat(:,:,k);    
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