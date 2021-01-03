% First experiment is assumed to be the reference case (no quantization)

function [echo_int_aver]=plot_quantization_data(filname,exptnum,necho,delt,pk)

clr={'b.-','r.-','k.-','m.-','b*-','r*-','k*-','m*-'};

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
    
    plot(tvect,echo_shape(:,i)/max(echo_shape(:,1)),clr{mod(i-1,4)+1}); hold on;
end
set(gca,'FontSize',14);
xlabel('Time (\mus)');
ylabel('Normalized echo shape');

figure(2); clf;
echo_int_aver=zeros(1,length(exptnum));
for i=1:length(exptnum)
    echo_int_aver(i)=mean(echo_int(tmp(2)/2:tmp(2),i));
end
echo_int_aver=echo_int_aver/echo_int_aver(1);

plot(delt,echo_int_aver,'bo-');
set(gca,'FontSize',14);
xlabel('Quantization time (\mus)');
if pk
    ylabel('Normalized echo peak');
else
    ylabel('Normalized echo area');
end