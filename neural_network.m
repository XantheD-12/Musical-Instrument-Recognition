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
    
    % NEWFF Create a feed-forward backpropagation network:创建一个前馈反向传播网络
    % NEWFF(P,T,S,TF,BTF,BLF,PF,IPF,OPF,DDF)
    % P：输入参数矩阵。(RxQ1)，其中Q1代表R元的输入向量。其数据意义是矩阵P有Q1列，每一列都是一个样本，而每个样本有R个属性（特征）。
    % 一般矩阵P需要归一化，即P的每一行都归一化到[0 1]或者[-1 1]。
    % T：目标参数矩阵。(SNxQ2)，Q2代表SN元的目标向量。
    % S：N-1个隐含层的数目（S（i）到S（N-1）），默认为空矩阵[]。输出层的单元数目SN取决于T。返回N层的前馈BP神经网络
    % TF：相关层的传递函数，默认隐含层为tansig函数，输出层为purelin函数。
    % BTF：BP神经网络学习训练函数，默认值为trainlm函数。
    % BLF：权重学习函数，默认值为learngdm。
    % PF：性能函数，默认值为mse，可选择的还有sse，sae，mae，crossentropy。
    % IPF，OPF，DDF均为默认值即可。
    % 返回N层前馈反向传播网络
    % net-前馈反向传播网络
    net = newff(input_data, mapped_label_data, [30], {'tansig' 'logsig'}, 'trainlm', '', 'mse', {}, {}, 'dividerand');

    % INIT  Initialize a time series object with new time and data values
    % 用新的时间和数据值初始化一个时间序列对象
    net = init(net);

    % Custom parameters
    % 设置训练参数
    net.trainParam.epochs = 500;
    net.trainParam.lr = 0.1;
    net.trainParam.min_grad = 0;
    net.trainParam.max_fail = 150;
    
    % Train network
    % 训练网络
    % input_data-网络实际输入，mapped_label_data-网络应有输出
    
    [trained_net, stats] = train(net, input_data, mapped_label_data);
    % tranied_net:训练得到的网络
    
    %输入数据
    %a=[3.0;9.3;3.3;2.05;100;2.8;11.2;50];
    %将输入数据归一化
    %a=premnmx(input_data);
    %放入到网络输出数据
    %b=sim(net,a);
    %将得到的数据反归一化得到预测数据
    %c=postmnmx(b,mint,maxt);
    
    % 乐器list
    instrument=["banjo","cello","clarinet","guitar","oboe","piano","trumpet","violin"];
    k=777;
    % 使用训练后的网络进行测试并与其标签对比
    t=trained_net(input_data(:,k));
    [max_t,index_t]=max(t);
    fprintf('识别乐器：%s \n',instrument(index_t));
    real=mapped_label_data(:,k);
    [max_r,index_r]=max(real);
    fprintf('乐器：%s \n',instrument(index_r));
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