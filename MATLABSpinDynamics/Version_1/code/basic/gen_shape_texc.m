% Generate pulse shape file in Bruker TOPSPIN format from variable time
% segment data
% amplitude -> normalized to 1, phase -> in degrees
% ntot -> number of segments

function gen_shape_texc(filname,ntot,timing,amplitude,phase)

power=1e2*amplitude.^2; % In percent
phase=mod(phase,360); % Phase modulo 2*pi (360 degrees)

% Open file
%filname_path=['wave/' filname]; % Unix
filname_path=['wave\' filname]; % Windows
fhandle1=fopen(filname_path,'w');

% Calculate segment lengths, correct for round-off errors
nseg=round(ntot*timing/sum(timing));
ntot_actual=sum(nseg);
disp(['Actual number of segments = ' num2str(ntot_actual)])
%nseg(length(power))=nseg(length(power))+(ntot-ntot_actual);

% Write header

fprintf(fhandle1,'%s\n',['##TITLE= /opt/PV4.0.0/exp/stan/nmr/lists/wave/' filname]);
fprintf(fhandle1,'%s\n','##JCAMP-DX= 5.00 Bruker JCAMP library');
fprintf(fhandle1,'%s\n','##DATA TYPE= Shape Data');
fprintf(fhandle1,'%s\n','##ORIGIN= Bruker Analytik GmbH');
fprintf(fhandle1,'%s\n','##OWNER= <nmrsu>');
fprintf(fhandle1,'%s\n','##DATE= 10/08/17');
fprintf(fhandle1,'%s\n','##TIME= 16:57:41');
fprintf(fhandle1,'%s\n',['##MINX= ' num2str(min(power))]);
fprintf(fhandle1,'%s\n',['##MAXX= ' num2str(max(power))]);
fprintf(fhandle1,'%s\n',['##MINY= ' num2str(min(phase))]);
fprintf(fhandle1,'%s\n',['##MAXY= ' num2str(max(phase))]);
fprintf(fhandle1,'%s\n','##$SHAPE_EXMODE= Inversion');
fprintf(fhandle1,'%s\n', '##$SHAPE_TOTROT= 1.800000e+02');
fprintf(fhandle1,'%s\n', '##$SHAPE_BWFAC= 1.116000e+00');
fprintf(fhandle1,'%s\n','##$SHAPE_INTEGFAC= 1.000000e+00');
fprintf(fhandle1,'%s\n', '##$SHAPE_MODE= 0');
fprintf(fhandle1,'%s\n', ['##NPOINTS= ' num2str(ntot)]);
fprintf(fhandle1,'%s\n','##XYPOINTS= (XY..XY)');

% Write (power,phase) pairs
for i=1:length(power)
    for j=1:nseg(i)
        fprintf(fhandle1,'%s\n',[num2str(power(i)) ', ' num2str(phase(i))]);
    end
end

% Close file
fclose(fhandle1);