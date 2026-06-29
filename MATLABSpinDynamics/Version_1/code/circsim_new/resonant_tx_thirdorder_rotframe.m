% Calculate coil current in rotating frame for use in spin dynamics
% calculations

function [out] = resonant_tx_thirdorder_rotframe(params,data)

tf = data.tf; I_L = data.I_L;

tfi = linspace(min(tf),max(tf),length(tf)); del_tfi = (tfi(2)-tfi(1)); % Uniform time vector
I_Li = interp1(tf,I_L,tfi); % Interpolated coil current

lo = exp(-1i*tfi); % Local oscillator
I_Lb = lo.*I_Li; % Rotating frame coil current

wn = del_tfi/pi; % Normalized RF frequency (sampling frequency = 2)
[b,a] = butter(4,0.7*wn); % Butterworth LPF, cutoff at 0.7*(RF frequency)
[gd,~] = grpdelay(b,a); % Group delay of Butterworth LPF
grd = gd(1)*del_tfi; % Group delay at low frequencies

I_Lbf = filter(b,a,I_Lb); % Filtered rotating frame coil current

% Calculate average rotating frame current (half-cycle average)
nseg = ceil((max(tfi)-min(tfi))/pi);
I_Lr = zeros(1,nseg); tr = I_Lr;
for i = 1:nseg
    ind = find((tfi-min(tfi))/pi > (i-1) & (tfi-min(tfi))/pi <= i);
    I_Lr(i) = mean(I_Lbf(ind)); tr(i) = mean(tfi(ind));
end
tr = tr - grd; % Compensate for filter group delay

% Throw away points before the pulse starts
ind = find(tr > min(tfi));
tr = tr(ind); I_Lr = I_Lr(ind); 

% Create output structure
out.tr = tr; out.I_Lr = I_Lr; 

% Plot results
if params.plt_tx;
    figure(2);
    plot(tr/(2*pi),real(I_Lr),'b.-'); hold on;
    plot(tr/(2*pi),imag(I_Lr),'r.-');
end

