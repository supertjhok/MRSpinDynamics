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
% Use 2rd order Trotter approximatation for relaxation during pulses 10/01/11
% Use Rodrigues' rotation formula to speed up simulation (~4x) 10/04/11
% Allow arbitrary w0, w1 maps 10/04/11

function [macq]=sim_spin_dynamics_arb5(tp,phi,amp,offs,acq,len_acq,T1,T2,del_w0,w_1)

T_max=1e6; % Maximum T1,T2 for relaxation to be considered
relax=min(T1,T2)<T_max; % Relaxation flag

numpts=length(del_w0);
if length(w_1)~=numpts
    disp('Error: w0 and w1 vectors have different lengths!');
    return;
end

% Calculate spectra (no diffusion)
m0=1; % Initial magnetization vector amplitude
onevect=ones(1,numpts);

mvect=zeros(3,numpts); % Magnetization vectors
mvect(3,:)=m0*onevect; % Initial mag vectors are along z-axis
mlong=zeros(3,numpts);

acq_cnt=0; % Acquisition counter
nacq=sum(acq); % Number of acquisitions
macq=zeros(nacq,numpts);

% window function for acquisition
window = sinc(del_w0*len_acq/(2*pi));
window = window./sum(window);

% Evolution of magnetization
num_pulses=length(phi);
for j=1:num_pulses
    if amp(j)>0 % RF pulse
        dw=-offs(j)+del_w0; w1=amp(j)*w_1;
        Omega=sqrt(w1.^2+dw.^2);
        w1=w1./Omega; dw=dw./Omega;
        
        neff=[w1*cos(phi(j)) % Calculate effective rotation axis
            w1*sin(phi(j))
            dw];
        theta=tp(j)*Omega; % Rotation angle
        
        if relax
            expT1=exp(-0.5*tp(j)/T1)*onevect;
            expT2=exp(-0.5*tp(j)/T2)*onevect;
            
            mlong(3,:)=m0*(1-expT1); % Longitudinal recovery
            
            mrelax=[expT2
                expT2
                expT1]; % Relaxation vector
            
            % Approximation for in-pulse relaxation
            mvect=mvect.*mrelax+mlong;
        end
        
        % Rodrigues' rotation formula
        cr=cross(neff,mvect); dt=dot(neff,mvect);
        cs=cos(theta); sn=sin(theta); cs1=1-cs;
        mvect=[mvect(1,:).*cs+cr(1,:).*sn+neff(1,:).*dt.*cs1
            mvect(2,:).*cs+cr(2,:).*sn+neff(2,:).*dt.*cs1
            mvect(3,:).*cs+cr(3,:).*sn+neff(3,:).*dt.*cs1];
        
        if relax
            % Approximation for in-pulse relaxation
            mvect=mvect.*mrelax+mlong;
        end
        
    else % Free precession
        Omega=-offs(j)+del_w0;
        theta=tp(j)*Omega; % Rotation angle
        
        % Rotation about z-axis
        cs=cos(theta); sn=sin(theta);
        mvect=[mvect(1,:).*cs-mvect(2,:).*sn
            mvect(1,:).*sn+mvect(2,:).*cs
            mvect(3,:)];
        
        if relax
            expT1=exp(-tp(j)/T1)*onevect;
            expT2=exp(-tp(j)/T2)*onevect;
            
            mlong(3,:)=m0*(1-expT1); % Longitudinal recovery
            
            mrelax=[expT2
                expT2
                expT1]; % Relaxation vector
            
            % Relaxation (exact)
            mvect=mvect.*mrelax+mlong;
        end
    end
    
    if acq(j) % Acquire spectrum
        acq_cnt=acq_cnt+1;
        trans=mvect(1,:)+1i*mvect(2,:); % Mx + iMy = +1 coherence
        macq(acq_cnt,:)=conv(trans,window,'same'); % Convolve with acquisition window
    end
end