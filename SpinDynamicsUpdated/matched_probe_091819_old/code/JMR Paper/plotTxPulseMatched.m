function [pulseOut] = plotTxPulseMatched(sp,pp)
%PLOTTXPULSEMATCHED Summary of this function goes here
%   Detailed explanation goes here

% Read in parameters


T_90=pp.T_90; % Nominal T_90 pulse length


% Set plotting parameters
sp.plt_tx = 1; sp.plt_rx = 0; sp.plt_sequence = 0; % Plots on/off
sp.plt_axis = 0; sp.plt_mn = 0; sp.plt_echo = 0;
sp.plt_output = 1; sp.plt_fields = 0;


% Design impedance matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Create pulse sequence (in normalized time)
% --------------------------------------------
% Pre-calculate all pulses for speed
pp_in.tp=pp.T_90; 
pp_in.phi=pi/2; 
pp_in.amp=1;
pp_in.tdel=200e-6; 
pp_out=calc_pulse_shape(sp,pp,pp_in); % Exc pulse y (phase = pi/2)

pulseOut = pp_out;

end

function [pp_out]=calc_pulse_shape(sp,pp,pp_in)

    T_90=pp.T_90;
    tdeln=(pi/2)*pp_in.tdel/T_90; % Normalized delay
    amp_zero=pp.amp_zero; % Minimum amplitude for calculations

    % Add delay to RF pulse to account for ring down, create structure
    pp_curr=pp;
    pp_curr.tp = [pp.preDelay pp_in.tp pp.postDelay];
    pp_curr.phi = [0 pp_in.phi 0];
    pp_curr.amp = [0 pp_in.amp 0];

    % Calculate RF pulse
    sp.plt_rx=0; % Turn off plotting
    [tvect, Icr, tf1, tf2] = find_coil_current(sp,pp_curr);
    pp_out.tf1=tf1; pp_out.tf2=tf2;

    delt=(pi/2)*(tvect(2)-tvect(1))/T_90; % Convert to normalized time
    texc=delt*ones(1,length(tvect));
    pexc=atan2(imag(Icr),real(Icr));
    aexc=abs(Icr);
    aexc(aexc<amp_zero)=0; % Threshold amplitude

    % Remove added delay from RF pulse
    pp_out.tp=[texc -tdeln]; pp_out.phi=[pexc 0]; pp_out.amp=[aexc 0];
    pp_out.acq=zeros(1,length(texc)+1);
end