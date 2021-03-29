% Plot performance of optimized refocusing pulses in time-varying fields
% with simple rectangular and RP2 pulses as a function of the amplitude of
% the field fluctuation.
% No probe effects are considered.
% --------------------------------------------------------------
% Written by: Soumyajit Mandal, 03/22/21

function compare_cpmg_results_ideal_tv(file,pulse_num)

% Fluctuation amplitude vector
B_0t_max=3; npts=16;
B_0t_amp=linspace(0,B_0t_max,npts); % Fluctuation amplitudes

% Load the results file
filname = file;

tmp=load(filname); results_all=tmp.results;
results=results_all{pulse_num};
sp=results{6}; pp=results{7};

% Define time vectors
w_1n=(pi/2)/pp.T_90; % Nominal nutation frequency
tacq=w_1n*pp.tacq; % Normalized acquisition window length
tdw=w_1n*pp.tdw; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
tvect=linspace(-tacq/2,tacq/2,nacq)'; % Acquisition vector, size: [nacq,1]

% Set plotting parameters
sp.plt_axis=0;  sp.plt_tx=0; sp.plt_rx=0; sp.plt_output=0;

% Define refocusing cycle
tp=results{1}; phi=results{2}; amp=results{3};
tE=sum(tp)/w_1n; % Echo period
pp.tref=tp/w_1n; pp.pref=phi; pp.aref=amp;

% Simulate optimized pulse for different field fluctuation strengths
B_0t_nom=sp.B_0t; % Nominal fluctuation
echo_rx_1=zeros(npts,nacq);
parfor i=1:npts
    sp_curr=sp;
    sp_curr.B_0t=B_0t_amp(i)*B_0t_nom;
    [~,echo_rx_1(i,:),~]=sim_cpmg_ideal_tv_final(sp_curr,pp);
end

% Plot asymptotic echoes
figure(1);
subplot(1,2,1);
imagesc(tvect,B_0t_amp,real(echo_rx_1)); colorbar;
set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
xlabel('t_{acq}/T_{90}'); ylabel('Fluctuation amplitude (norm)');
title('Re(s_{rx})');

subplot(1,2,2);
imagesc(tvect,B_0t_amp,imag(echo_rx_1)); colorbar;
set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
xlabel('t_{acq}/T_{90}'); ylabel('Fluctuation amplitude (norm)');
title('Im(s_{rx})');

% Simulate standard pulses under the same conditions
echo_rx_2=zeros(npts,nacq); echo_rx_3=echo_rx_2; echo_rx_4=echo_rx_2;
parfor i=1:npts
    sp_curr=sp; pp_curr=pp;
    sp_curr.B_0t=B_0t_amp(i)*B_0t_nom;
    
    % Rect-180
    tfp=(tE-pp_curr.T_180)/2;
    pp_curr.tref=[tfp pp_curr.T_180 tfp]; pp_curr.pref=[0 0 0]; pp_curr.aref=[0 1 0];
    [~,echo_rx_2(i,:),~]=sim_cpmg_ideal_tv_final(sp_curr,pp_curr);
    
    % Rect-135
    tfp=(tE-0.75*pp_curr.T_180)/2;
    pp_curr.tref=[tfp 0.75*pp_curr.T_180 tfp]; pp_curr.pref=[0 0 0]; pp_curr.aref=[0 1 0];
    [~,echo_rx_3(i,:),~]=sim_cpmg_ideal_tv_final(sp_curr,pp_curr);
    
    % RP2-1.0
    tfp=(tE-pp_curr.T_180)/2;
    pp_curr.tref=[tfp pp_curr.T_180*[0.14 0.72 0.14] tfp];
    pp_curr.pref=[0 pi*[1 0 1] 0]; pp_curr.aref=[0 [1 1 1] 0];
    [~,echo_rx_4(i,:),~]=sim_cpmg_ideal_tv_final(sp_curr,pp_curr);
end

% Reference scans with no fluctuations)
%pp.NE=pp.NEmin; % No need to simulate many echoes
sp.B_0t=zeros(1,pp.NE); % No field fluctuations

pp.tref=tp/w_1n; pp.pref=phi; pp.aref=amp;
[~,echo_rx_ref_1,~]=sim_cpmg_ideal_tv_final(sp,pp);
norm_1=sqrt(trapz(tvect,abs(echo_rx_ref_1).^2));
echo_rx_ref_1=(echo_rx_ref_1)'/norm_1; 
% Note: ' denotes conjugate transpose

tfp=(tE-pp.T_180)/2;
pp.tref=[tfp pp.T_180 tfp]; 
pp.pref=[0 0 0]; pp.aref=[0 1 0];
[~,echo_rx_ref_2,~]=sim_cpmg_ideal_tv_final(sp,pp);
norm_2=sqrt(trapz(tvect,abs(echo_rx_ref_2).^2));
echo_rx_ref_2=(echo_rx_ref_2)'/norm_2;

tfp=(tE-0.75*pp.T_180)/2;
pp.tref=[tfp 0.75*pp.T_180 tfp]; 
pp.pref=[0 0 0]; pp.aref=[0 1 0];
[~,echo_rx_ref_3,~]=sim_cpmg_ideal_tv_final(sp,pp);
norm_3=sqrt(trapz(tvect,abs(echo_rx_ref_3).^2));
echo_rx_ref_3=(echo_rx_ref_3)'/norm_3;

tfp=(tE-pp.T_180)/2;
pp.tref=[tfp pp.T_180*[0.14 0.72 0.14] tfp]; 
pp.pref=[0 pi*[1 0 1] 0]; pp.aref=[0 [1 1 1] 0];
[~,echo_rx_ref_4,~]=sim_cpmg_ideal_tv_final(sp,pp);
norm_4=sqrt(trapz(tvect,abs(echo_rx_ref_4).^2));
echo_rx_ref_4=(echo_rx_ref_4)'/norm_4;

% Find echo amplitudes after matched filtering
echo_mf_1=zeros(npts,1); % Size: [npts,1]
echo_mf_2=echo_mf_1; echo_mf_3=echo_mf_1; echo_mf_4=echo_mf_1;
for i=1:npts
    echo_mf_1(i)=trapz(tvect,echo_rx_1(i,:).*echo_rx_ref_1); % Estimate echo rms
    echo_mf_2(i)=trapz(tvect,echo_rx_2(i,:).*echo_rx_ref_2); % Estimate echo rms
    echo_mf_3(i)=trapz(tvect,echo_rx_3(i,:).*echo_rx_ref_3); % Estimate echo rms
    echo_mf_4(i)=trapz(tvect,echo_rx_4(i,:).*echo_rx_ref_4); % Estimate echo rms
end

% Normalize amplitudes to Rect-180
echo_mf_1=echo_mf_1./norm_2; echo_mf_2=echo_mf_2./norm_2;
echo_mf_3=echo_mf_3./norm_2; echo_mf_4=echo_mf_4./norm_2;

figure(2);
plot(B_0t_amp,real(echo_mf_1),'LineWidth',1); hold on;
plot(B_0t_amp,real(echo_mf_2),'LineWidth',1);
plot(B_0t_amp,real(echo_mf_3),'LineWidth',1);
plot(B_0t_amp,real(echo_mf_4),'LineWidth',1);
set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
xlabel('Fluctuation amplitude (\Delta\omega_{0,max}/\omega_{1n})'); ylabel('s_{mf} (normalized)');
legend({'Optimized','Rect-180','Rect-135','RP2-1.0'});
