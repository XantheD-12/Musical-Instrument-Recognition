fprintf('start!\n');
% �������ݼ���������
dataset="allset";
n_chunks = 50;
% [input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'only_attack');
% [input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'without_attack');
%[input_data, mapped_label_data]=process_training_data(dataset,n_chunks,'');
%"dataset_chunk50"
%"dataset_chunk50_only_attack"
%"dataset_chunk50_without_attack"
%����������������Ϊ.mat�ļ�
[input_data, mapped_label_data] = process_training_data(dataset, n_chunks, 'without_attack');
save("dataset_chunk50_1",'input_data','');
