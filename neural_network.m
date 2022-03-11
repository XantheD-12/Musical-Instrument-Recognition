% Load processed dataset
 load 'dataset_chunk50.mat'
% load 'dataset_chunk50_only_attack.mat'
% load 'dataset_chunk50_without_attack.mat'
% load 'dataset_chunk50_1.mat'

% Base Experiment
%load 'datasets/1-dataset_avgvalue_chunk50.mat'

% Only Attack
% load 'datasets/2-dataset_avgvalue_chunk50_only_attack.mat'

% Without Attack
% load 'datasets/3-dataset_avgvalue_chunk50_without_attack.mat'

% 100 Hz
% load 'datasets/4-dataset_avgvalue_chunk50_100hz.mat'

% 900 Hz
% load 'datasets/5-dataset_avgvalue_chunk50_900hz.mat'

% ---------------------------------------------------------------------

% Functions used to process the data, prelimiarily not available in this
% repo.
% dataset = 'dataset-2/';
% n_chunks = 50;
% [input_data, mapped_label_data] = process_training_data(dataset, n_chunks, '');
% [input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'only_attack');
% [input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'without_attack');

% ---------------------------------------------------------------------

n_runs = 1;
sum_error = 0;

for n = 1:n_runs
    %fprintf('Running experiment %d of %d', n, n_runs); 

    % Rprop & Early stopping
    
    % NEWFF Create a feed-forward backpropagation network:����һ��ǰ�����򴫲�����
    % NEWFF(P,T,S,TF,BTF,BLF,PF,IPF,OPF,DDF)
    % P�������������(RxQ1)������Q1����RԪ�����������������������Ǿ���P��Q1�У�ÿһ�ж���һ����������ÿ��������R�����ԣ���������
    % һ�����P��Ҫ��һ������P��ÿһ�ж���һ����[0 1]����[-1 1]��
    % T��Ŀ���������(SNxQ2)��Q2����SNԪ��Ŀ��������
    % S��N-1�����������Ŀ��S��i����S��N-1������Ĭ��Ϊ�վ���[]�������ĵ�Ԫ��ĿSNȡ����T������N���ǰ��BP������
    % TF����ز�Ĵ��ݺ�����Ĭ��������Ϊtansig�����������Ϊpurelin������
    % BTF��BP������ѧϰѵ��������Ĭ��ֵΪtrainlm������
    % BLF��Ȩ��ѧϰ������Ĭ��ֵΪlearngdm��
    % PF�����ܺ�����Ĭ��ֵΪmse����ѡ��Ļ���sse��sae��mae��crossentropy��
    % IPF��OPF��DDF��ΪĬ��ֵ���ɡ�
    % ����N��ǰ�����򴫲�����
    % net-ǰ�����򴫲�����
    net = newff(input_data, mapped_label_data, [30], {'tansig' 'logsig'}, 'trainlm', '', 'mse', {}, {}, 'dividerand');

    % INIT  Initialize a time series object with new time and data values
    % ���µ�ʱ�������ֵ��ʼ��һ��ʱ�����ж���
    net = init(net);

    % Custom parameters
    % ����ѵ������
    net.trainParam.epochs = 500;
    net.trainParam.lr = 0.1;
    net.trainParam.min_grad = 0;
    net.trainParam.max_fail = 150;
    
    % Train network
    % ѵ������
    % input_data-����ʵ�����룬mapped_label_data-����Ӧ�����
    
    [trained_net, stats] = train(net, input_data, mapped_label_data);
    % tranied_net:ѵ���õ�������
    
    %��������
    %a=[3.0;9.3;3.3;2.05;100;2.8;11.2;50];
    %���������ݹ�һ��
    %a=premnmx(input_data);
    %���뵽�����������
    %b=sim(net,a);
    %���õ������ݷ���һ���õ�Ԥ������
    %c=postmnmx(b,mint,maxt);
    
    % ����list
    instrument=["banjo","cello","clarinet","guitar","oboe","piano","trumpet","violin"];
    k=777;
    % ʹ��ѵ�����������в��Բ������ǩ�Ա�
    t=trained_net(input_data(:,k));
    [max_t,index_t]=max(t);
    fprintf('ʶ��������%s \n',instrument(index_t));
    real=mapped_label_data(:,k);
    [max_r,index_r]=max(real);
    fprintf('������%s \n',instrument(index_r));
    t2=input_data;
    t2=sim(trained_net,t2);
    
    plotconfusion(mapped_label_data, sim(trained_net, input_data))
    [c, cm] = confusion(mapped_label_data, sim(trained_net, input_data));
    sum_error = sum_error + c;
    fprintf('  %f%% accuracy \n', (1 - c)*100);
    %save("model",'trained_net');
end

avg_accuracy = (n_runs - sum_error) / n_runs;
fprintf('\nAverage accuracy: %f%% \n', avg_accuracy*100);