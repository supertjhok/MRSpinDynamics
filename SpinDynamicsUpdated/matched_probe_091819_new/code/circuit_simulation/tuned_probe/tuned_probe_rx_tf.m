function [tf] = tuned_probe_rx_tf(sp,pp)

    k=sp.k;
    T=sp.T;
    L=sp.L;
    R=sp.R; 
    C=sp.C;
    Cin=sp.Cin; 
    Rin=sp.Rin; 
    Rd=sp.Rd;
    vn=sp.vn;
    in=sp.in;

    w0=sp.w0; % Larmor frequency
    del_w=sp.del_w;
    w1_max=(pi/2)/pp.T_90;

    s=1i*(w0+del_w*w1_max);
    f=imag(s)/(2*pi); % Un-normalized frequency axis
    Yin=s*Cin+1/Rin; % Input admittance
    Yp=s*C+1/Rd+1./(s*L+R); % Admittance of parallel resonator

    tf=1./(1+(s*L+R).*(s*C+1/Rd+Yin));
    Zs=1./(Yin+Yp);

    % Calculate receiver waveform and noise PSD
    tf=tf.*(2*pi*f/w0).^2;
end
