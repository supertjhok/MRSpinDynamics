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
% Allow relaxation within pulses (based on Bain et al., JMR 2010) 09/30/11

function [macq,del_w]=sim_spin_dynamics_arb(tp,phi,amp,offs,acq,len_acq,T1,T2)

% Uniform distribution of del_w0 (uniform gradient)
maxoffs=20;
numpts=2e3+1; % number of spin vectors to simulate

% Resonance offset distribution
del_w=linspace(-maxoffs,maxoffs,numpts);
numpts=length(del_w);

% Calculate spectra (no relaxation during pulses or diffusion)
% Magnetization: [Mx, My, Mz, Me]
m0=1; % Initial magnetization vector amplitude
mvect=zeros(4,1,numpts); % Magnetization vectors
mvect(3,1,:)=m0*ones(1,1,numpts); % Initial mag vectors are along z-axis
mvect(4,1,:)=mvect(3,1,:); % Equilibrium magnetization

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
        
    % Propagate magnetization
    mvect=prop_mag(mvect,del_w+offs(j),amp(j),tp(j),phi(j),T1,T2,numpts);
    
    if acq(j) % Acquire spectrum
        acq_cnt=acq_cnt+1;
        for k=1:numpts
            trans(k)=mvect(1,1,k)+1i*mvect(2,1,k); % +1 coherence = Mx + iMy
        end
        
        fy = conv(trans,window);
        macq(acq_cnt,:) = fy(((numpts+1)/2:3*(numpts-1)/2+1));
    end
end


% Propagate magnetization in time, including relaxation
function mvect_out = prop_mag(mvect,del_w,w1,tp,phi,T1,T2,numpts)

R2=1/T2; R1=1/T1;

mvect_out=mvect;
for k=1:numpts
    % Evolution matrix
    A=[-R2 -del_w(k) w1*sin(phi) 0
        del_w(k) -R2 -w1*cos(phi) 0
        -w1*sin(phi) w1*cos(phi) -R1 R1
        0 0 0 0];
    
    %mvect_out(:,:,k)=expm(A*tp)*mvect(:,:,k); % Most robust, slow
    %mvect_out(:,:,k)=expmdemo3(A*tp)*mvect(:,:,k); % Usually 2x faster
    mvect_out(:,:,k)=padm(A*tp)*mvect(:,:,k); % Intermediate speed
    %mvect_out(:,:,k)=expv(tp,A,mvect(:,:,k)); % Very slow
end
