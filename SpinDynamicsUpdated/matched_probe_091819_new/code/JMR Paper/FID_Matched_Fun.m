function [mrx,echo_rx,tvect,SNR] = FID_Matched_Fun( sp,pp )


% Define pulse system parameters
T_90=pp.T_90; % Nominal T_90 pulse length

% Create receiver sensitivity map
% sp.w_1r=sp.w_1; % , assuming single transmit/receive coil


% Set acquisition parameters
tacq=(pi/2)*pp.tacq/T_90; % Normalized acquisition window length
tdw=(pi/2)*pp.tdw/T_90; % Normalized receiver dwell time
nacq=round(tacq/tdw)+1; % Number of acquired time domain points
tvect=linspace(-tacq/2,tacq/2,nacq)'; % Acquisition vector, size: [nacq,1]
isoc=exp(1i*tvect*sp.del_w); % Isochromats (without additional gradients)

% Design impedance matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Create pulse sequence (in normalized time)
% --------------------------------------------
% Pre-calculate all pulses for speed
Rtot={};
pp_in.tp=pp.T_90;
pp_in.tdel=2*pp.T_90;
pp_in.phi=pi/2;
pp_in.amp=1;
pp_out=calc_pulse_shape(sp,pp,pp_in); % Exc pulse y (phase = pi/2)
sp.tf1=pp_out.tf1; 
sp.tf2=pp_out.tf2; % Save Rx transfer functions
Rtot{1}=calc_rotation_matrix(sp,pp_out);

% Excitation pulses 1 and 2, including timing correction
texc=[pi/2 1e9 10e9];
aexc=[1 0 0];
pexc1=[1 0 0]; 
acq_exc=[0 0 1];
gexc=[0 0 0]; % gexc is the gradient
pul1=[pexc1]; % Sequence 1: (y,x)

% Create complete pulse sequences
% Fixed terms
pp.tp=[texc];
pp.amp=[aexc];
pp.acq=[acq_exc];
pp.grad=[gexc];
pp.Rtot=Rtot;

% Calculate gradient strength vectors
% Create fields and sample parameters
% sp=create_fields_single_sided(sp);
% Estimate maximum gradient frequency offsets using specified FOV

spc=sp;
spc.del_wg = 0;
ppc = pp;
        % Run PAP phase cycle: (1-2)
        ppc.pul=pul1; 
       [macq,mrx,pnoise,f,echo_rx,tvect,mvect]=calc_macq_matched_probe_relax3(spc,ppc);
        figure
     plot((real((macq))))
     hold on
     plot((imag((macq))))
     title('Macq')
     
     figure
     plot((real(mrx)));
     hold on
     plot((imag((mrx))));
     title('MRX')
      
%         ppc.pul=pul2; 
%         [~,mrx2]=calc_macq_matched_probe_relax4(spc,ppc);
%         mrx_x=(mrx1-mrx2); % Apply phase cycle
        
        
        % Calculate time-domain echoes
%         echo_rx_x=isoc*mrx1'; % Size: [nacq,NE]
%         echo_rx_y=isoc*mrx_y'; % Size: [nacq,NE]
%         echo_rx_xy=imag(echo_rx_x)-1i*real(echo_rx_y); % Raw echo shapes
%         echo_int_all(i,j,:)=trapz(tvect,echo_rx_xy)'; % Estimate echo integrals, size: [NE,1]
%         
SNR  = 1;
  sp.plt_output = 0;
if sp.plt_output % Plot selected outputs
    eplt=2; % Echo number to plot
    echo_int_eplt=echo_int_all(:,:,eplt); % Select data for chosen echo
    
    figure; % Plot k-space of selected echo
    subplot(1,3,1); imagesc(real(echo_int_eplt));
    colorbar; title('k-space (Real)')
    subplot(1,3,2); imagesc(imag(echo_int_eplt));
    colorbar; title('k-space (Imag)')
    subplot(1,3,3); imagesc(abs(echo_int_eplt));
    colorbar; title('k-space (Mag)')
    
    figure; % Plot spin density image
    img_eplt=ifftshift(ifft2(echo_int_eplt)); % Estimate image
    imagesc(abs(img_eplt));
    colorbar; title('Spin density image (Mag)');
end
end

function [pp_out]=calc_pulse_shape(sp,pp,pp_in)

T_90=pp.T_90;
tdeln=(pi/2)*pp_in.tdel/T_90; % Normalized delay
amp_zero=pp.amp_zero; % Minimum amplitude for calculations

% Add delay to RF pulse to account for ring down, create structure
pp_curr=pp;
pp_curr.tp = [pp_in.tp pp_in.tdel];
pp_curr.phi = [pp_in.phi 0];
pp_curr.amp = [pp_in.amp 0];

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
pp_out.tp=[texc -tdeln]; 
pp_out.phi=[pexc 0];
pp_out.amp=[aexc 0];
pp_out.acq=zeros(1,length(texc)+1);
end

