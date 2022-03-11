fprintf('start!\n');
% 根据数据集生成特征
dataset="allset";
n_chunks = 50;
% [input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'only_attack');
% [input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'without_attack');
%[input_data, mapped_label_data]=process_training_data(dataset,n_chunks,'');
%"dataset_chunk50"
%"dataset_chunk50_only_attack"
%"dataset_chunk50_without_attack"
%生成特征，并保存为.mat文件
[input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'without_attack');
save("dataset_chunk50_1",'input_data','');
