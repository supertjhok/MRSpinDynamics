% Simulate a CPMG-IR sequence with rectangular pulses and finite
% transmit and receive bandwidth
% -------------------------------------------------------------------

% NE -> Number of echoes to simulate
% TE -> Echo spacing (real value)
% tauvect -> Initial encoding vector (s)
% T1, T2 -> Relaxation time constants (s)
function [echo_int_all,echo_asymp_all]=sim_cpmg_ir_matched_probe_relax(NE,TE,tauvect,T1,T2)



% Simulate each echo of a CPMG sequence
[sp, pp]=set_params_matched; % Define system parameters
T_90=pp.T_90; % Nominal T_90 pulse length

% Set plotting parameters
sp.plt_tx = 0; sp.plt_rx = 0; sp.plt_sequence = 0; % Plots /off
sp.plt_axis = 0; sp.plt_mn = 0; sp.plt_echo = 0;

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Create (w0, w1) and coil sensitivity maps
numpts=sp.numpts; maxoffs=10;
sp.del_w=linspace(-maxoffs,maxoffs,numpts); % Linear gradient
sp.w_1=ones(1,numpts); % Uniform transmit w_1
sp.w_1r=ones(1,numpts); % Uniform receiver sensitivity

% Set sample parameters
sp.m0=ones(1,numpts); sp.mth=ones(1,numpts); % Spin density
sp.T1=T1*ones(1,numpts); sp.T2=T2*ones(1,numpts); % Relaxation constants

% Set acquisition parameters
tacq=(pi)*pp.tacq/T_90; % Normalized acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points

% Create pulse sequence (in real time)
% --------------------------------------------
ntau=length(tauvect); % Number of encoding steps
echo_int_all=zeros(ntau,NE); % Output echo integrals
echo_asymp_all=zeros(ntau,NE,nacq);
'Starting spin dynamics simulation'
parfor j=1:ntau % Parallelize for speed
    ppc=pp; % Local variable
    
    % Encoding period
    tenc=[ppc.T_180 tauvect(j)];
    penc=[0 0];
    aenc=[1 0];
    acq_enc=[0 0];
    
    % Excitation pulse, including timing correction
    texc=[1 -2/pi]*ppc.T_90;
    pexc=[pi/2 0];
    aexc=[1 0];
    acq_exc=[0 0];
    
    % Refocusing cycles
    numComp = 3
    tref=zeros(1,numComp*NE); 
    pref=tref; 
    aref=tref; 
    acq_ref=tref;
    for i=1:NE
        tref((i-1)*3+1:i*3)=[((TE-ppc.T_180)/2) ppc.T_180 ((TE-ppc.T_180)/2)];
        pref((i-1)*3+1:i*3)= [0 0 0];
        aref((i-1)*3+1:i*3)=[0 1 0];
        acq_ref((i-1)*3+1:i*3)=[0 0 1];
    end
    
    % Create complete pulse sequence
    ppc.tp=[tenc texc tref]; ppc.phi=[penc pexc pref];
    ppc.amp=[aenc aexc aref]; ppc.acq=[acq_enc acq_exc acq_ref];
    
    % Run PAP phase cycle
    [~,mrx1,pnoise,f,~,~]=calc_macq_matched_probe_relax(sp,ppc);
    
    ppc.phi=[penc pexc+pi pref]; % Change phase of excitation pulse
    [~,mrx2,~,~,~,~]=calc_macq_matched_probe_relax(sp,ppc);
    
    mrx=(mrx1-mrx2); pnoise=pnoise*sqrt(2); % Apply phase cycle
    
    % Calculate time-domain echoes
    % Can't plot inside a parfor loop
     
     echo_int_ind = zeros(1,NE);
     echo_rx=zeros(NE,nacq);
    for i=1:NE
        
        [echo_rx(i,:),tvect]=calc_time_domain_echo_arb(mrx(i,:),sp.del_w,tacq,tdw,sp.plt_echo);         
        % Calculate echo integral
       echo_int_ind(i) = trapz(tvect,echo_rx(i,:))
    end
    echo_asymp_all(j,:,:) = echo_rx;
    echo_int_all(j,:)=echo_int_ind;
    j
end

% Plot results
figure;
subplot(1,2,1); imagesc(TE*linspace(1,NE,NE)*1e3,tauvect*1e3,real(echo_int_all)); colorbar;
set(gca,'FontSize',14); xlabel('Relaxation time (ms)'); ylabel('Encoding time (ms)');
title('Real');
subplot(1,2,2); imagesc(TE*linspace(1,NE,NE)*1e3,tauvect*1e3,imag(echo_int_all)); colorbar;
set(gca,'FontSize',14); xlabel('Relaxation time (ms)'); ylabel('Encoding time (ms)');
title('Imag');
set(gca,'FontSize',14); xlabel('Relaxation time (ms)'); ylabel('Encoding time (ms)'); 

figure;
for i = 1:ntau
    plot(TE*linspace(1,NE,NE)*1e3,imag(echo_int_all(i,:)));
    hold on
end

