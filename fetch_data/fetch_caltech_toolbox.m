
cur_dir = pwd;
cd(fileparts(mfilename('fullpath')));

try
    fprintf('Downloading caltech toolbox ...\n');
    urlwrite('http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/code/code3.2.1.zip', ...
        'code3.2.1.zip');

    fprintf('Unzipping...\n');
    unzip('code3.2.1.zip', '../external/code3.2.1');

    fprintf('Done.\n');
    delete('code3.2.1.zip');
catch
    fprintf('Error in downloading'); 
end

cd(cur_dir);

