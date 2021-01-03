% Use Martin/Yi-Qiao's terminology to simulate dynamics of spin-1/2
% ensembles with precalculated refocusing matrix, refocusing and excitation
% pulses have arbitrary power levels
% ------------------------------------
% texc, tref = pulse durations (sec)
% pexc, pref = phases in radians
% aexc, aref = normalized pulse amplitudes = 0 for free precession
% t_acq = acquisition length (sec)
% ne = number of echoes
% del_w = vector of resonance frequency offsets
% all coherence pathways are considered in this simulation, and all echoes
% are saved
% ------------------------------------
% Soumyajit Mandal 08/26/10
% Allowed arbitrary pulse amplitudes 02/25/11
% Code clean up 03/07/11

function [mecho]=sim_spin_dynamics_arba_echoes(texc,pexc,aexc,...
    tref,pref,aref,ne,del_w,t_acq)

% Initial magnetization vector amplitude
m0=1;

% Calculate echo spectrum (no relaxation or diffusion)
numpts=length(del_w);
mvect=zeros(3,1,numpts); % Magnetization vectors
mvect(1,1,:)=m0*ones(1,1,numpts); % Initial mag vectors are along z-axis

% Excitation
num_exc=length(pexc);
for j=1:num_exc
    if aexc(j)>0
        w1=aexc(j);
        Omega=sqrt(w1.^2+del_w.^2);
        
        mat=calc_matrix_elements(del_w,w1,Omega,texc(j),pexc(j)); % RF pulses
    else
        mat=calc_fp_matrix_elements(del_w,texc(j)); % Free precession
    end
    for k=1:numpts
        mvect(:,:,k)=mat(:,:,k)*mvect(:,:,k);
    end
end

my=zeros(1,numpts); mecho=zeros(ne,numpts);
cp=map_pi(+1); cm=map_pi(-1);

% window function for acquisition only between the 180 pulses
window = sinc(del_w*t_acq/(2*pi));
window = window/sum(window);

% Refocusing
num_ref=length(pref)/ne; % Number of elements in each refocusing cycle
for l=1:ne % Number of refocusing cycles
    for j=1:num_ref
        ind=j+(l-1)*num_ref;
        if aref(ind)>0
            w1=aref(ind);
            Omega=sqrt(w1.^2+del_w.^2);
            
            mat=calc_matrix_elements(del_w,w1,Omega,tref(ind),pref(ind)); % RF pulses
        else
            mat=calc_fp_matrix_elements(del_w,tref(ind)); % Free precession
        end
        
        if j==num_ref
            for k=1:numpts
                mvect(:,:,k)=mat(:,:,k)*mvect(:,:,k);
                % Only My is visible
                my(k)=(mvect(cp,1,k)-mvect(cm,1,k))/(2*1i); % My;
            end
        else
            for k=1:numpts
                mvect(:,:,k)=mat(:,:,k)*mvect(:,:,k);
            end
        end
    end
    
    fy = conv(my,window);
    mecho(l,:) = fy(((numpts+1)/2:3*(numpts-1)/2+1));
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