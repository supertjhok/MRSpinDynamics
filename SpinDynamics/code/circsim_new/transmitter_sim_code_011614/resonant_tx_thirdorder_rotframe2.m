% Calculate coil current in rotating frame for use in spin dynamics
% calculations
% Assume original vector is uniformly spaced

function [out] = resonant_tx_thirdorder_rotframe2(params,data)

tf = data.tf; I_L = data.I_L;

lo = exp(-1i*tf); % Local oscillator
I_Lb = lo.*I_L; % Rotating frame coil current

wn = params.wn; % Normalized RF frequency (sampling frequency = 2)
b = params.btr_b; a = params.btr_a; % Butterworth LPF
grd = params.grd; % Group delay of LPF

I_Lbf = filter(b,a,I_Lb); % Filtered rotating frame coil current

% Calculate average rotating frame current (half-cycle average)
nseg = ceil((max(tf)-min(tf))/pi);
I_Lr = zeros(1,nseg); tr = I_Lr;
for i = 1:nseg
    ind = find((tf-min(tf))/pi > (i-1) & (tf-min(tf))/pi <= i);
    I_Lr(i) = mean(I_Lbf(ind)); tr(i) = mean(tf(ind));
end
tr = tr - grd; % Compensate for filter group delay

% Throw away points before the pulse starts
ind = find(tr > min(tf));
tr = tr(ind); I_Lr = I_Lr(ind); 

% Create output structure
out.tr = tr; out.I_Lr = I_Lr; 

% Plot results
if params.plt_tx;
    figure(2);
    plot(tr/(2*pi),real(I_Lr),'b.-'); hold on;
    plot(tr/(2*pi),imag(I_Lr),'r.-');
end

