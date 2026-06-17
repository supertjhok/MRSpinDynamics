% /dsk/670/hurliman/pulsecraft/DIFFUSION/PATHWAYS/coherences.m

% This program calculates the contributions of all coherence pathways contributing to the CPMG sequence for the 2nd to
% Nfinal-th echo and the diffusive attenuation rate. It is assumed that the
% echoes are filtered with the asymptotic echo shape. For details, see
% JMR148, 367-378 (2001)

% Modified by Soumyajit Mandal, 03/04/11
% Bugfixes, 03/07/11
% CPMG sequence parameters
% texc, pexc, aexc -> excitation pulse segment lengths (sec), phases (radians),
% and normalized amplitudes
% tref, pref, aref -> refocusing pulse segment lengths (sec), phases (radians),
% and normalized amplitudes
% A pulse of amplitude = 1 results in w1 = 1
% T_E -> echo spacing (sec)

% Nfinal -> number of echoes to calculate
% The higher the echo number, the longer the calculation will be. Nfinal = 10 is just a few seconds,
% but Nfinal = 15 will take several minutes.

function [axistime,disttimes]=...
    coherences_new(texc,pexc,aexc,tref,pref,aref,T_E,Nfinal)

% calculate the asymptotic echo shape for a CPMG sequence in a gradient
% field without diffusion.
[masy,omega0]=asy_new(texc,pexc,aexc,tref,pref,aref,T_E);

if length(texc)==1e2 % Van's excitation pulse
    masy = real(masy); % Mx
else
    masy = -imag(masy); % My
end
del_w0=omega0(2)-omega0(1);
normasy = sqrt(sum(masy.*masy)/del_w0);

% calculate the transition elements between the difference coherences for
% excitation and refocusing pulses
% and for the initial 90 pulse
[Lambda,Linit]=lambdas_new(texc,pexc,aexc,tref,pref,aref,omega0);

msums=zeros(Nfinal,length(omega0));
disttimes = zeros(Nfinal,3*64);
axistime = 10.^(-(-1:length(disttimes)-2)./64);
numberpaths=zeros(Nfinal,1);

for iecho = 2:Nfinal
    Necho = iecho - 1;
    % number of full periods
    % there are (Necho +1) 180 pulses and this calculates the effect of the
    % (Necho +1)th echo.
    
    disp(iecho)
    numberpath = 0;
    
    for Ntrans = 0:Necho  % change number of full periods
        % in the transverse plane
        
        %%%%%% calculate all different ways to distribute the Ntrans
        %%%%%% periods among the Necho periods
        [pointertrans,~,itrans]=marbles_new(Necho,Ntrans);
        
        %%%%%% calculate all different ways to distribute the
        %%%%%% +1 and -1 coherences among the Ntrans periods
        [pointerplus,pointerminus,imax]=marbles_new(Ntrans,fix(Ntrans/2));
        
        %%%%%% calculate all different coherence pathways that add up to zero
        coherence = zeros(imax*itrans,Necho);
        
        for icase = 1:itrans
            for icoh = 1:imax
                iout = (icase-1)*imax+icoh;
                coherence(iout,pointertrans(icase,pointerplus(icoh,:))) = 1;
                coherence(iout,pointertrans(icase,pointerminus(icoh,:)))=-1;
            end
        end
        
        % for practical reasons, change coherence by adding 2,
        %     i.e. -1  goes to 1
        %           0          2
        %          +1          3
        coherence = coherence + 2;
        
        iinit = 1 + rem( Ntrans,2);  %calculate whether first coherence is -1 or +1
        
        for ipath = 1:imax*itrans
            
            % calculate the weight for each path by matched filtering with asy. echo
            ilambda1 = (iinit-1)*6+coherence(ipath,1);
            ilambdae = 3*(coherence(ipath,Necho)-1)+3;
            ilambdam = 3.*(coherence(ipath,1:Necho-1)-1)+coherence(ipath,2:Necho);
            ilambda  = [ilambda1 ilambdam ilambdae]';
            if length(texc)==1e2 % Van's excitation pulse
                mpathy = real(Linit(iinit,:).*prod(Lambda(ilambda,:)));
            else
                mpathy = imag(Linit(iinit,:).*prod(Lambda(ilambda,:)));
            end
            wpath  = sum(mpathy.*masy)./normasy;
            
            % calculate the spectrum of the sum of all pathways (should approach masy)
            msums(iecho,:)   = mpathy + msums(iecho,:);
            
            % calculate the diffusion decay time for this path
            a1k = cumsum([0 (coherence(ipath,:)-2).*(0.5+(0:Necho-1)) 0.25+Necho/2]);
            qt  = [iinit-1.5 (coherence(ipath,:)-2) 0.5];
            a2k = -[(0:1:Necho)+0.5 (Necho+1)].*cumsum(qt);
            diffrate = (Ntrans+0.25)/3 + qt*(a1k'+a2k');
            % normalized decay time w.r.t. to on resonance
            difftime = (Necho+1)/(12*diffrate);
            
            indextime = min( 3*64, floor(-64*log10(difftime))+2 );
            disttimes(iecho,indextime) = disttimes(iecho,indextime) + wpath;
            % hold on
            % plot(omega0,mpathy,'r')
        end
        
        numberpath = numberpath + imax*itrans;
    end
    
    numberpaths(Necho+1) = numberpath;
    
    %     subplot(2,1,1)
    %       hold off
    %       plot(omega0,msums(iecho,;),'r')
    %       hold on
    %       plot(omega0,masy,'b')
    %       xlabel('\omega_0 / \omega_1')
    %       ylabel('M (\omega_0) / M_o')
    %       title(['Comparison of spectrum of ',num2str(iecho),'-th echo with asymptotic spectrum'])
    
    %     subplot(2,1,2)
    %       semilogx(axistime,disttimes(iecho,;))
    %       axis([0.8e-3 1.2 -0.2 1])
    %
    
end

disp(numberpaths')

% Comparison of asymptotic echo shape (in frequency domain) with
% contributions from all coherence pathways for the last echo
% calculated. For the later echoes, we expect them to coincide to a
% large degree
figure(10)

for iecho = 2:Nfinal
    subplot(5,3,iecho)
    hold off
    plot(omega0,msums(iecho,:),'r')
    hold on
    plot(omega0,masy,'b')
    xlim([min(omega0) max(omega0)])
    text(-9.2, 1,[num2str(iecho),'-th echo'],'fontsize',8)
    %xlabel('\omega_0 / \omega_1')
    %ylabel('M (\omega_0) / M_o')
end
for iecho = Nfinal-2:Nfinal
    subplot(5,3,iecho)
    xlabel('\omega_0 / \omega_1')
end

% Distribution of diffusive attenuation rate for Nfinal-th echo
figure(11)
eta_l=1./axistime;
bar(eta_l,disttimes(Nfinal,:))
set(gca,'xscale','log')
axisold =axis;
axis([0.5 250 axisold(3:4)])
xlabel('\eta_l^{(N)}')
ylabel('a_l^{(N)}')
title(['N = ',num2str(Nfinal)])

eta_av=-log(sum(disttimes(Nfinal,:).*exp(-eta_l))/sum(disttimes(Nfinal,:))) % mean decay rate