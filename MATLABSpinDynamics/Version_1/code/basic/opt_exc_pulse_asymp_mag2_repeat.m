% Repeated optimization runs, store current maximum
function opt_exc_pulse_asymp_mag2_repeat(nseg,tref,pref,aref,delt)

count=1;
[neff,del_w]=calc_rot_axis_arba(tref,pref,aref);

results=[];
while(1)
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_asymp_mag2(nseg,neff,del_w,delt,[]);
    if count==1
        echo_rms_max=echo_rms;
        echo_pk_max=echo_pk;
        save results_mag2_pk.mat texc pexc echo_pk echo_rms
        save results_mag2_rms.mat texc pexc echo_pk echo_rms
    else
        if echo_rms>echo_rms_max
            echo_rms_max=echo_rms;
            save results_mag2_rms.mat texc pexc echo_pk echo_rms
        end
        if echo_pk>echo_pk_max
            echo_pk_max=echo_pk;
            save results_mag2_pk.mat texc pexc echo_pk echo_rms
        end
    end
    
    results(count,1:nseg)=texc;
    results(count,nseg+1:2*nseg)=pexc;
    results(count,2*nseg+1)=echo_pk;
    results(count,2*nseg+2)=echo_rms;
    save results_mag2.mat results
    
    disp(count)
    disp(echo_pk_max)
    disp(echo_rms_max)
    count=count+1;
    
    tmp=input('Press any key to continue','s')
end
