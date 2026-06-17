% Repeated optimization runs, store current maximum
function opt_exc_ref_pulses_repeat(nexc,nref,T_90,NE,T_FP,delt)

count=1;

results=[];
while(1)
    [texc,pexc,tref,pref,echo_pk,echo_rms]=opt_exc_ref_pulses(nexc,nref,T_90,NE,T_FP,delt); % Fixed phases
    if count==1
        echo_rms_max=echo_rms;
        echo_pk_max=echo_pk;
        save exc_pulse_results_pk.mat texc pexc tref pref echo_pk echo_rms
        save exc_pulse_results_rms.mat texc pexc tref pref echo_pk echo_rms
    else
        if echo_rms>echo_rms_max
            echo_rms_max=echo_rms;
            save exc_pulse_results_rms.mat texc pexc tref pref echo_pk echo_rms
        end
        if echo_pk>echo_pk_max
            echo_pk_max=echo_pk;
            save exc_pulse_results_pk.mat texc pexc tref pref echo_pk echo_rms
        end
    end
    
    results(count,1:nexc)=texc;
    results(count,nexc+1:2*nexc)=pexc;
    results(count,2*nexc+1:2*nexc+nref)=tref;
    results(count,2*nexc+nref+1:2*(nexc+nref))=pref;
    results(count,2*(nexc+nref)+1)=echo_pk;
    results(count,2*(nexc+nref)+2)=echo_rms;
    save results_exc_ref.mat results
    
    disp(count)
    disp(echo_pk_max)
    disp(echo_rms_max)
    count=count+1;
end