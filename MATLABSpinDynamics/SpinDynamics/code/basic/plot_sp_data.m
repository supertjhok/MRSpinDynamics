function [echo_int,echo_shape_orig,echo_shape]=plot_sp_data(filname,exptnum,necho,grad,pk)

clr={'bo-','rv-','kd-','m^-'};

% Note: the results are quite insensitive to the particular value of
% nignore. For simplicity it can be set to zero
nignore=4; % Ignore first few echoes (CPMG transients)
nwidth=28; % Number of points to integrate per echo
irat=4;  % Interpolation factor to get smoother echo shapes

figure(1); clf;
for i=1:length(exptnum)
    
    [data,parameter] = readbrukerfile(filname,exptnum(i));
    tmp=size(data);  del=tmp(1)/necho;
    nignore_ind=1+tmp(1)*nignore/necho;
    
    dw=parameter.dw;
    
    tvect_orig=dw*linspace(0,del-1,del)'*1e6; % us
    tvect=dw*linspace(0,del-1,del*irat)'*1e6; % us
    
    if i==1
        echo_shape_orig=zeros(del,length(exptnum));
        echo_shape=zeros(del*irat,length(exptnum));
        echo_int=zeros(tmp(2),length(exptnum));
    end
    
    data=abs(data);
    
    for k=1:tmp(2)
        
        echo_shape_orig(:,i)=zeros(del,1);
        echo_shape(:,i)=zeros(del*irat,1);
        
        for j=1:necho-nignore
            ind=nignore_ind+del*(j-1):nignore_ind+del*j-1;
            echo_shape_orig(:,i)=echo_shape_orig(:,i)+data(ind,k);
        end
        echo_shape(:,i)=interp1(tvect_orig,echo_shape_orig(:,i),tvect,'spline');
        
        if pk
            % Find the peak of all echoes
            echo_int(k,i)=max(echo_shape_orig(:,i));
        else
            % Find the SNR of all echoes
            ind=(del*irat-nwidth)/2:(del*irat+nwidth)/2-1;
            echo_int(k,i)=trapz(echo_shape(ind,i).^2)/nwidth; % Matched filter
        end
    end
    
    plot(grad,echo_int(:,i),clr{mod(i-1,4)+1}); hold on;
end
xlabel('Gradient (G/cm)');
if pk
    ylabel('Echo peak');
else
    ylabel('Echo area');
end


figure(2); clf;
if strcmp(filname,'cpmg_oneshot_sp_2') || strcmp(filname,'cpmg_oneshot_sp_3')
    plot(grad,echo_int(:,10)./echo_int(:,9),'kd-'); % RP2-1.0 - 0.5 ms
    hold on;
    plot(grad,echo_int(:,4)./echo_int(:,3),'ko-'); % RP2-1.0 - 1 ms
    plot(grad,echo_int(:,2)./echo_int(:,1),'k*-');  % RP2-1.0 - 2 ms
    
    plot(grad,echo_int(:,7)./echo_int(:,3),'k^-'); % RP2-1.0b - 1 ms
    plot(grad,echo_int(:,8)./echo_int(:,3),'kv-'); % RP2-1.0c - 1 ms
    
    plot(grad,echo_int(:,5)./echo_int(:,3),'bo-'); % RP2-1.3 - 1 ms
    plot(grad,echo_int(:,6)./echo_int(:,3),'ro-'); % RP2-1.9 - 1 ms
end

if strcmp(filname,'cpmg_oneshot_sp_4')
    plot(grad,echo_int(:,10)./echo_int(:,1),'bo-'); % hard pulse - 1.4 ms (different run)
    hold on;
    plot(grad,echo_int(:,2)./echo_int(:,1),'rs-'); % RP2-1.0a - 1.4 ms
    %plot(grad,echo_int(:,3)./echo_int(:,1),'k^-'); % RP2-1.0b - 1.4 ms
    %plot(grad,echo_int(:,4)./echo_int(:,1),'kv-');  % RP2-1.0c - 1.4 ms
    
    plot(grad,echo_int(:,5)./echo_int(:,1),'kd-'); % RP2-1.3 - 1.4 ms
    plot(grad,echo_int(:,6)./echo_int(:,1),'mv-'); % RP2-1.9 - 1.4 ms
    
    %plot(grad,echo_int(:,9)./echo_int(:,1),'k.-'); % hard pulse - 1.4 ms (different run, not shaped)
    
    legend({'Rectangular','RP2-1.0','RP2-1.3','RP2-1.9'});
end

if strcmp(filname,'cpmg_oneshot_sp_5')
    plot(grad,echo_int(:,2)./echo_int(:,1),'ko-'); % rectangular, RP2-1.0a
    hold on;
    plot(grad,echo_int(:,3)./echo_int(:,1),'k^-'); % optimal, rectangular
    plot(grad,echo_int(:,4)./echo_int(:,1),'kv-'); % optimal, RP2-1.0a
    
    plot(grad,echo_int(:,5)./echo_int(:,1),'k.-'); % rectangular, rectangular (different run)
    legend({'rectangular, RP2-1.0a','optimal, rectangular','optimal, RP2-1.0a','rectangular, rectangular (different run)'});
end

if strcmp(filname,'cpmg_oneshot_sp_6') || strcmp(filname,'cpmg_oneshot_sp_7')
    %plot(grad,echo_int(:,2)./echo_int(:,2),'bo-'); % rectangular, rectangular (corrected)
    %plot(grad,echo_int(:,3)./echo_int(:,2),'r^-'); % CP-M1, RP2-1.0a
    %plot(grad,echo_int(:,5)./echo_int(:,2),'rd-');  % CP-M3, RP2-1.0a
    %plot(grad,echo_int(:,6)./echo_int(:,2),'rs-');  % CP-M4, RP2-1.0a
    %plot(grad,echo_int(:,8)./echo_int(:,2),'k.-'); % VAN_EXC_AMPL, RP2-1.0a
    
    plot(grad,echo_int(:,4)./echo_int(:,2),'rv-');  % CP-M2, RP2-1.0a
    hold on;
    plot(grad,echo_int(:,9)./echo_int(:,2),'r.-');  % CP-M5, RP2-1.0a
    plot(grad,echo_int(:,10)./echo_int(:,2),'r^-');  % CP-M6, RP2-1.0a
    plot(grad,echo_int(:,11)./echo_int(:,2),'rs-');  % CP-M8, RP2-1.0a
    
    plot(grad,echo_int(:,7)./echo_int(:,2),'k*-'); % VAN_EXC, RP2-1.0a
    legend({'CP-M2, RP2-1.0a','CP-M5, RP2-1.0a','CP-M6, RP2-1.0a','CP-M8, RP2-1.0a','VAN-EXC, RP2-1.0a'});
end

if strcmp(filname,'cpmg_oneshot_sp_8')
    plot(grad,echo_int(:,2)./echo_int(:,1),'rv-');  % CP-M6, RP2-1.0a
    hold on;
    plot(grad,echo_int(:,3)./echo_int(:,1),'r.-');  % CP-M8, RP2-1.0a
    %plot(grad,echo_int(:,4)./echo_int(:,1),'r^-');  % CP-M9, RP2-1.0a
    plot(grad,echo_int(:,5)./echo_int(:,1),'rs-');  % CP-M10, RP2-1.0a
    plot(grad,echo_int(:,6)./echo_int(:,1),'rd-');  % CP-M12, RP2-1.0a
    %legend({'CP-M6, RP2-1.0a','CP-M8, RP2-1.0a','CP-M9, RP2-1.0a','CP-M10, RP2-1.0a','CP-M12, RP2-1.0a'});
    legend({'CP-M6, RP2-1.0a','CP-M8, RP2-1.0a','CP-M10, RP2-1.0a','CP-M12, RP2-1.0a'});
end

set(gca,'FontSize',14);
xlabel('Gradient (G/cm)');
if pk
    ylabel('Normalized echo peak');
else
    ylabel('Normalized echo area');
end

figure(3); clf;
if strcmp(filname,'cpmg_oneshot_sp_2') || strcmp(filname,'cpmg_oneshot_sp_3')
    plot(tvect,echo_shape(:,10)/max(echo_shape(:,9)),'kd-'); % RP2-1.0 - 0.5 ms
    hold on;
    plot(tvect,echo_shape(:,4)/max(echo_shape(:,3)),'ko-'); % RP2-1.0 - 1 ms
    plot(tvect,echo_shape(:,2)/max(echo_shape(:,1)),'k*-');  % RP2-1.0 - 2 ms
    
    plot(tvect,echo_shape(:,7)/max(echo_shape(:,3)),'k^-'); % RP2-1.0b - 1 ms
    plot(tvect,echo_shape(:,8)/max(echo_shape(:,3)),'kv-'); % RP2-1.0c - 1 ms
    
    plot(tvect,echo_shape(:,5)/max(echo_shape(:,3)),'bo-'); % RP2-1.3 - 1 ms
    plot(tvect,echo_shape(:,6)/max(echo_shape(:,3)),'ro-'); % RP2-1.9 - 1 ms
end

if strcmp(filname,'cpmg_oneshot_sp_4')
    plot(tvect,echo_shape(:,2)/max(echo_shape(:,1)),'ko-'); % RP2-1.0a - 1.4 ms
    hold on;
    plot(tvect,echo_shape(:,3)/max(echo_shape(:,1)),'k^-'); % RP2-1.0b - 1.4 ms
    plot(tvect,echo_shape(:,4)/max(echo_shape(:,1)),'kv-');  % RP2-1.0c - 1.4 ms
    
    plot(tvect,echo_shape(:,5)/max(echo_shape(:,1)),'bo-'); % RP2-1.3 - 1.4 ms
    plot(tvect,echo_shape(:,6)/max(echo_shape(:,1)),'ro-'); % RP2-1.9 - 1.4 ms
    
    %plot(tvect,echo_shape(:,9)/max(echo_shape(:,1)),'k.-'); % hard pulse - 1.4 ms (different run, not shaped)
    plot(tvect,echo_shape(:,10)/max(echo_shape(:,1)),'r.-'); % hard pulse - 1.4 ms (different run)
    legend({'RP2-1.0a','RP2-1.0b','RP2-1.0c','RP2-1.3','RP2-1.9','Hard pulse (different run)'});
end

if strcmp(filname,'cpmg_oneshot_sp_5')
    plot(tvect,echo_shape(:,2)/max(echo_shape(:,1)),'ko-'); % rectangular, RP2-1.0a
    hold on;
    plot(tvect,echo_shape(:,3)/max(echo_shape(:,1)),'k^-'); % optimal, rectangular
    plot(tvect,echo_shape(:,4)/max(echo_shape(:,1)),'kv-'); % optimal, RP2-1.0a
    
    plot(tvect,echo_shape(:,5)/max(echo_shape(:,1)),'k.-'); % rectangular, rectangular (different run)
    legend({'rectangular, RP2-1.0a','optimal, rectangular','optimal, RP2-1.0a','rectangular, rectangular (different run)'});
end

if strcmp(filname,'cpmg_oneshot_sp_6') || strcmp(filname,'cpmg_oneshot_sp_7')
    %plot(tvect,echo_shape(:,3)/max(echo_shape(:,2)),'r^-'); % CP-M1, RP2-1.0a
    %plot(tvect,echo_shape(:,5)/max(echo_shape(:,2)),'rd-');  % CP-M3, RP2-1.0a
    %plot(tvect,echo_shape(:,6)/max(echo_shape(:,2)),'rs-');  % CP-M4, RP2-1.0a
    %plot(tvect,echo_shape(:,8)/max(echo_shape(:,2)),'k.-'); % %VAN_EXC_AMPL, RP2-1.0a
    
    plot(tvect,echo_shape(:,2)/max(echo_shape(:,2)),'bo-'); % rectangular, rectangular (corrected)
    hold on;
    plot(tvect,echo_shape(:,4)/max(echo_shape(:,2)),'rv-');  % CP-M2, RP2-1.0a
    plot(tvect,echo_shape(:,9)/max(echo_shape(:,2)),'r.-');  % CP-M5, RP2-1.0a
    plot(tvect,echo_shape(:,10)/max(echo_shape(:,2)),'r^-');  % CP-M6, RP2-1.0a
    plot(tvect,echo_shape(:,11)/max(echo_shape(:,2)),'rs-');  % CP-M8, RP2-1.0a
    
    plot(tvect,echo_shape(:,7)/max(echo_shape(:,2)),'k*-'); % VAN_EXC, RP2-1.0a
    legend({'rectangular, rectangular','CP-M2, RP2-1.0a','CP-M5, RP2-1.0a','CP-M6, RP2-1.0a','CP-M8, RP2-1.0a','VAN-EXC, RP2-1.0a'});
end

if strcmp(filname,'cpmg_oneshot_sp_8')
    plot(tvect,echo_shape(:,1)/max(echo_shape(:,1)),'bo-'); % rectangular, rectangular (corrected)
    hold on;
    plot(tvect,echo_shape(:,2)/max(echo_shape(:,1)),'rv-');  % CP-M6, RP2-1.0a
    plot(tvect,echo_shape(:,3)/max(echo_shape(:,1)),'r.-');  % CP-M8, RP2-1.0a
    %plot(tvect,echo_shape(:,4)/max(echo_shape(:,1)),'r^-');  % CP-M9, RP2-1.0a
    plot(tvect,echo_shape(:,5)/max(echo_shape(:,1)),'rs-');  % CP-M10, RP2-1.0a
    plot(tvect,echo_shape(:,6)/max(echo_shape(:,1)),'rd-');  % CP-M12, RP2-1.0a
    %legend({'rectangular, rectangular','CP-M6, RP2-1.0a','CP-M8, RP2-1.0a','CP-M9, RP2-1.0a','CP-M10, RP2-1.0a','CP-M12, RP2-1.0a'});
    legend({'rectangular, rectangular','CP-M6, RP2-1.0a','CP-M8, RP2-1.0a','CP-M10, RP2-1.0a','CP-M12, RP2-1.0a'});
end

set(gca,'FontSize',14);
xlabel('Time (\mus)');
ylabel('Normalized echo shape');