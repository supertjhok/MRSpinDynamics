% Plot results of refocusing pulse optimization in time-varying fields
% --------------------------------------------------------------
% params = [tref, pref, aref, tfp, tacq, Rs(off,on)] (all times normalized to w1 = 1)
% --------------------------------------------------------------

function [neff,SNR] = plot_opt_ref_results_ideal_tv(file,pulse_num)

% Load the results file
filname = file;

tmp=load(filname); results_all=tmp.results;
siz = size(results_all);

% Plot optimization result (axis_rms) of all pulses for comparison
axis_rms=zeros(1,siz(1));
for i=1:siz(1)
tmp=results_all{i}; axis_rms(i)=tmp{4};
end
figure(11); plot(axis_rms,'LineWidth',1); hold on;
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
ylabel('Optimized SNR (rms)');
    
results=results_all{pulse_num};
SNR=results{4};
plot(pulse_num,SNR,'rs','MarkerSize',10);
params=results{5};
sp=results{6}; pp=results{7};

sp.plt_axis=1;  sp.plt_tx=1; sp.plt_rx=1; % Set plotting parameters

w_1n=(pi/2)/pp.T_90; % Nominal nutation frequency
% Plot refocusing cycle
tp=results{1}; phi=results{2}; amp=results{3}; 
pp.tref=tp/w_1n; pp.pref=phi; pp.aref=amp;
del_w=sp.del_w; plt=1;
[neff,alpha]=calc_rot_axis_arba4(tp,phi,amp,del_w,plt);
[v0crit]=calc_v0crit(del_w,neff,alpha,plt);

figure(13);
stairs([0 cumsum(tp/w_1n)]*1e6,[amp.*cos(phi) 0],'LineWidth',1); hold on;
stairs([0 cumsum(tp/w_1n)]*1e6,[amp.*sin(phi) 0],'LineWidth',1);
xlabel('Time (\mus)'); ylabel('Complex amplitude');
set(gca,'FontSize',14); set(gca,'FontWeight','Bold');

% Define time vectors
NE=pp.NE; % Number of echoes
tacq=w_1n*pp.tacq; % Normalized acquisition window length
tdw=w_1n*pp.tdw; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
tvect=linspace(-tacq/2,tacq/2,nacq)'; % Acquisition vector, size: [nacq,1]
tE=sum(pp.tref); % Echo period
tseq=tE*linspace(0.5,NE-0.5,NE); % Actual time

% Plot main scan
sp.plt_output=1; %sp.B_0t=sp.B_0t*1.5;
[~,echo_rx,~]=sim_cpmg_ideal_tv(sp,pp);

% Plot reference scan
pp.NE=pp.NEmin;
sp.B_0t=zeros(1,pp.NE); sp.plt_output=0;
[~,echo_rx_ref,~]=sim_cpmg_ideal_tv_final(sp,pp);
echo_rx_ref=(echo_rx_ref)'/sqrt(trapz(tvect,abs(echo_rx_ref).^2));
% Note: ' denotes conjugate transpose

% Find echo amplitudes after matched filtering
echo_mf=zeros(NE,1); % Size: [NE,1]
for i=1:NE
    echo_mf(i)=trapz(tvect,echo_rx(i,:).*echo_rx_ref); % Estimate echo rms
end

figure(8);
plot(tseq*1e3,real(echo_mf),'LineWidth',1); hold on;
plot(tseq*1e3,imag(echo_mf),'LineWidth',1);
set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
xlabel('Sequence time (ms)'); ylabel('s_{mf}(t)');
