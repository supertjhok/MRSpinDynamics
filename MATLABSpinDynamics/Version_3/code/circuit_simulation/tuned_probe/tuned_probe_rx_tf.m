function [tf] = tuned_probe_rx_tf(sp,pp)
% TUNED_PROBE_RX_TF
% Calculate the tuned-probe receive transfer function on the offset grid.
%
% Signature
%   tf = tuned_probe_rx_tf(sp,pp)
%
% Inputs
%   sp - System parameter structure with tuned-probe circuit values and the
%     normalized offset grid sp.del_w.
%   pp - Pulse parameter structure; pp.T_90 sets the nominal RF amplitude used
%     to convert normalized offsets to absolute angular frequency.
%
% Outputs
%   tf - Complex receive transfer function evaluated at w0 + del_w*w1_max.
%
% Notes
%   The final factor converts the resonator response to the voltage scaling
%   used by the receive-chain model.
% -------------------------------------------------------------------------

    L=sp.L;
    R=sp.R; 
    C=sp.C;
    Cin=sp.Cin; 
    Rin=sp.Rin; 
    Rd=sp.Rd;

    w0=sp.w0; % Larmor frequency
    del_w=sp.del_w;
    w1_max=(pi/2)/pp.T_90;

    s=1i*(w0+del_w*w1_max);
    f=imag(s)/(2*pi); % Un-normalized frequency axis
    Yin=s*Cin+1/Rin; % Input admittance

    tf=1./(1+(s*L+R).*(s*C+1/Rd+Yin));

    % Calculate receive-chain voltage scaling.
    tf=tf.*(2*pi*f/w0).^2;
end
