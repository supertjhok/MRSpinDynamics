% Use Martin/Yi-Qiao's terminology to simulate dynamics of spin-1/2
% ensembles with frequency offsets, refocusing and excitation
% pulses have arbitrary power levels
% ------------------------------------
% T1, T2 = relaxation times
% tp = durations in us
% phi = phases in radians
% amp = normalized amplitudes = 0 for free precession
% len_acq = acquisition length (us)
% acq = acquire signal if 1
% all coherence pathways are considered in this simulation
% ------------------------------------
% Soumyajit Mandal 08/26/10
% Allowed arbitrary pulse amplitudes 02/25/11
% Use normalized w1, allow relaxation during free precession 03/16/11
% Use approximatation for relaxation during pulses 10/01/11

function [macq,del_w]=sim_spin_dynamics_arb3(tp,phi,amp,offs,acq,len_acq,T1,T2)

cm=map_pi(-1); c0=map_pi(0); cp=map_pi(1);

% Uniform distribution of del_w0 (uniform gradient)
maxoffs=20;
numpts=1e4+1; % number of spin vectors to simulate

% Resonance offset distribution
%del_w=maxoffs*(2*rand(1,numpts)-1); % Random distribution
del_w=linspace(-maxoffs,maxoffs,numpts); % Deterministic distribution
numpts=length(del_w);

% Calculate spectra (no relaxation during pulses or diffusion)
m0=1; % Initial magnetization vector amplitude
mvect=zeros(3,1,numpts); % Magnetization vectors
mvect(c0,1,:)=m0*ones(1,1,numpts); % Initial mag vectors are along z-axis
mlong=zeros(3,1,numpts);
mrelax=ones(3,1,numpts);

acq_cnt=0; % Acquisition counter
nacq=sum(acq); % Number of acquisitions
macq=zeros(nacq,numpts);
trans=zeros(1,numpts);

% window function for acquisition only between the 180 pulses
window = sinc(del_w*len_acq/(2*pi));
window = window./sum(window);

% Evolution of magnetization
num_pulses=length(phi);
for j=1:num_pulses
    if amp(j)>0
        w1=amp(j);
        Omega=sqrt(w1.^2+(del_w+offs(j)).^2);
        mat=calc_matrix_elements(del_w+offs(j),w1,Omega,tp(j),phi(j)); % RF pulses
        
        mlong(c0,1,:)=m0*(1-exp(-0.5*tp(j)/T1))*ones(1,1,numpts); % Longitudinal recovery
        
        mrelax(c0,1,:)=exp(-0.5*tp(j)/T1)*ones(1,1,numpts); % Relaxation
        mrelax(cm,1,:)=exp(-0.5*tp(j)/T2)*ones(1,1,numpts); 
        mrelax(cp,1,:)=mrelax(cm,1,:);
        
        % Approximation for in-pulse relaxation
        mvect=mvect.*mrelax+mlong;
        
        %for k=1:numpts
        %    mvect(:,:,k)=mat(:,:,k)*mvect(:,:,k);
        %end
        mvect=multiprod(mat,mvect); 
        
        % Approximation for in-pulse relaxation
        mvect=mvect.*mrelax+mlong;
    else
        mat=calc_fp_matrix_elements(del_w,tp(j),T1,T2); % Free precession
        mlong(c0,1,:)=m0*(1-exp(-tp(j)/T1))*ones(1,1,numpts); % Longitudinal relaxation
        
        %for k=1:numpts
        %    mvect(:,:,k)=mat(:,:,k)*mvect(:,:,k)+mlong(:,:,k);
        %end
        mvect=multiprod(mat,mvect)+mlong;
    end
    
    if acq(j) % Acquire spectrum
        acq_cnt=acq_cnt+1;
        for k=1:numpts
            trans(k)=mvect(cp,1,k); % Only +1 coherence is visible
        end
        
        fy = conv(trans,window);
        macq(acq_cnt,:) = fy(((numpts+1)/2:3*(numpts-1)/2+1));
    end
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

% For free precession, w1 = 0, include relaxation
function R = calc_fp_matrix_elements(del_w,tf,T1,T2)

i=sqrt(-1);
R_pp=exp(-tf/T2)*(cos(del_w*tf)+i*sin(del_w*tf));
R_mm=conj(R_pp);
R_00=exp(-tf/T1)*ones(1,length(del_w));
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