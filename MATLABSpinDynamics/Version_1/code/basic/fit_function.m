function val=fit_function(pexc,texc,aexc,neff,del_w,len_acq)

[masy]=cpmg_van_spin_dynamics_asymp_mag2(texc,pexc,aexc,neff,del_w,len_acq);

% Calculate time-domain echo
%echo=zeros(1,length(tvect));
%for i=1:length(tvect)
%    echo(i)=sum(masy.*exp(-1i*del_w*tvect(i)));
%end

% Phase inversion leaves behind only the symmetric part of the spectrum,
% i.e., the real component of the time-domain echo
%echo = real(echo);

% Optimize echo peak
%val=-0.01*max(abs(echo));

% Optimize echo RMS + echo peak
%val=-0.01*(sqrt(trapz(tvect,abs(echo).^2))+max(abs(echo)));

masy=real(masy);
val=-sqrt(trapz(del_w,masy.^2))-0.33*trapz(del_w,masy);