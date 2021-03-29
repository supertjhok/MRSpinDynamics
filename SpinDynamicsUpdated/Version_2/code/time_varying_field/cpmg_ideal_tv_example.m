% Set parameters
[sp, pp]=set_params_ideal_tv;
sp.rho=1; sp.T1map=1e8; sp.T2map=1e8; % Set sample properties

% Create field and sample parameter vectors
sp=create_fields_lingrad(sp);
sp.del_wg=ones(1,length(sp.del_w)); % Create spatially uniform field offset vector

NE=pp.NE; % Number of echoes
w_1n=(pi/2)/pp.T_90; % Nominal nutation frequency
tE=sum(pp.tref); % Echo period
tseq=tE*linspace(0.5,NE-0.5,NE); % Actual time

tacq=w_1n*pp.tacq; % Normalized acquisition window length
tdw=w_1n*pp.tdw; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
tvect=linspace(-tacq/2,tacq/2,nacq)'; % Acquisition vector, size: [nacq,1]

% Define field fluctuation waveform
w_0max=2.0; % Maximum normalized frequency offset
% Constant
% sp.B_0t=zeros(1,NE);
% Linear ramp
% sp.B_0t=(w_1n*w_0max/sp.gamma)*linspace(0,1,NE);
% Triangle wave
%sp.B_0t=(w_1n*w_0max/sp.gamma)*[linspace(0,1,NE/4) linspace(1,-1,NE/2) linspace(-1,0,NE/4)];
sp.B_0t=(w_1n*w_0max/sp.gamma)*[linspace(0,1,NE/2) linspace(1,0,NE/2)];
% Sinusoid
% sp.B_0t=(w_1n*w_0max/sp.gamma)*sin(2*pi*linspace(0,1,NE));

% Plot refocusing cycle
tp=w_1n*pp.tref; phi=pp.pref; amp=pp.aref; del_w=sp.del_w; plt=1;
[n,alpha]=calc_rot_axis_arba4(tp,phi,amp,del_w,plt);

sp.plt_output = 1;
[mrx,echo_rx,echo_int]=sim_cpmg_ideal_tv(sp,pp);

% Run a reference scan with no field fluctuations
% Don't need too many echoes for this - just need the asymptotic shape
pp.NE=pp.NEmin; % Minimum number of echoes
sp.B_0t=zeros(1,pp.NE); sp.plt_output=0;
[mrx_ref,echo_rx_ref,~]=sim_cpmg_ideal_tv_final(sp,pp);
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
