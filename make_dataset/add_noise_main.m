%%% generate simulated Gaussian noisy frames for Vimeo-90K datasets

clear all;
clc;

sigma = 25;
data_path = 'I:/Datasets/vimeo_septuplet/sequences/';
noise_path = 'I:/Datasets/vimeo_septuplet/sequences_with_noise_25/';

if ~exist(noise_path, 'dir')
    mkdir(noise_path);
end

videos = dir(data_path);

total_num = 0;
for i = 3 : length(videos) %1~96
    seps = dir(fullfile(data_path, videos(i).name));
    seqs_num = length(seps)-2;
    total_num = total_num + seqs_num;
end


count = 0;

disp('Start adding noise to input...')
for i = 3 : length(videos)
    disp(['video num: ' num2str(i-2)])
    if strcmp(videos(i).name, '.') || strcmp(videos(i).name, '..')
        continue
    end
    
    seps = dir(fullfile(data_path, videos(i).name));
    for j = 1:length(seps)
        if strcmp(seps(j).name, '.') || strcmp(seps(j).name, '..')
            continue
        end
        add_noise_to_input(fullfile(data_path, videos(i).name, seps(j).name), ...
            fullfile(noise_path, videos(i).name, seps(j).name), sigma)
        count = count + 1;
        if mod(count,500)==0
            disp(['  Processed ',num2str(count/total_num*100), '%'])
        end
    end
end
disp(['  Processed ',num2str(count/total_num*100), '%'])






function add_noise_to_input(data_path, output_path, sigma)

filenames   = dir(fullfile(data_path, '*.png'));
num_imgs    = length(filenames);
img_list    = cell(num_imgs, 1);
if ~exist(output_path, 'dir')
    mkdir(output_path);
end
for iimg = 1 : num_imgs
    img_list{iimg} = imread([data_path sprintf('/im%d',iimg) '.png']);
    
    img_list{iimg} = addnoise(img_list{iimg}, sigma);
    imwrite(img_list{iimg}, [output_path '/im' sprintf('%04d',iimg) '.png']);
end

end


function y = addnoise(I, sigma)

I = double(I);
y = I + sigma * randn(size(I));
y = uint8(y);

end





