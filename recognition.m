% instrument recognition
% Charles Gao 2021-12-24
% 
% input: a wav file of music by instruments in this list:
% ["banjo","cello","clarinet","guitar","oboe","piano","trumpet","violin"]
%1
% the naming rule of the name of input file:
% "{instrument}_{tone}_{author}.wav"
%
% output: the recognition result and the supposed result

load model
% R = audiorecorder( 44100,16,2) ;
% flag=input('输入1开始录音(输入0停止录音):\n');
% if flag==1
%     record(R);
%     fprintf('-----开始录音-----\n')
% end
% while 1
%     flag=input('输入1开始录音(输入0停止录音):\n');
%     if flag==0
%         stop(R);
%         % 停止录制
%         myspeech = getaudiodata(R);
%         play(R);
%     end
% end
    
name = input('请输入文件名(输入-1停止):\n','s');
while ~strcmp(name,'-1')
      % file_name=strcat('./allset/',name);
     file_name = strcat('./inputs/',name);
    try
        % read the audio file
        [y,Fs] = audioread(file_name);
        % number of sampples
        Nsamps = length(y);
        % dircectory of the file
%         dir = file_name(1:9);
        % name of the file
        name = file_name(10:end);
        % FFT procedure
        y_fft = abs(fft(y));
        y_fft = y_fft(1:round(Nsamps/2)); % discard half of the points
        
        % processing the file name
        exploded_file_name = strsplit(name, '_');
        real = exploded_file_name(1);
        tone = exploded_file_name(2);
        % get the frequency difference
        frequency_diff = frequency_difference(tone{1});
        % shift frequency
        if isrow(y_fft)
             shifted_fft = shift_frequency(transpose(y_fft), frequency_diff, Fs, Nsamps);
        else
            shifted_fft = shift_frequency(y_fft, frequency_diff, Fs, Nsamps);
        end
        
        % limit shift_fft
        shifted_fft = shifted_fft(1:round(1000 * Nsamps / Fs));
        
        % extract the feature and get the input of the network
        feature_vector = create_feature_vector(shifted_fft,50);
        feature_vector = normalize_data(feature_vector);
        input_data = transpose(feature_vector);
        
        % invoke the network to recognize the instrument  
        instrument=["banjo","cello","clarinet","guitar","oboe","piano","trumpet","violin"];
        predicted = trained_net(input_data);
        [max_t,index_t]=max(predicted);
        fprintf('识别结果：%s \n',instrument(index_t));
        fprintf('预期结果：%s \n',real{1});
        name = input('请输入文件名(输入-1停止):\n','s');
    catch err
        fprintf('文件不存在\n');
        name = input('请输入文件名(输入-1停止):\n','s');
    end
end