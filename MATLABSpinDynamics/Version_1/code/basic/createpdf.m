% Create PDF files from all EPS files in a directory

function [fnames]=createpdf(folder)

% Get list of files
cmd=['dir /b ' folder '\*.eps'];
[status,result]=system(cmd);

len=size(result);
fnames={};

% Format list of files
count=0; pos=1;
for i=1:len(2)
    if double(result(i))==10 % LF character
        count=count+1;
        fnames{count}=result(pos:i-1);
        pos=i+1;
    end
end

% Convert EPS to PDF
for i=1:count
    cmd=['epstopdf ' folder '\' fnames{i}];
    [status,result]=system(cmd);
end
