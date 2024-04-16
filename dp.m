%% 清除指令
% clear 
% 
% 
% % Load the dataset 加载数据集
% addpath('DATA');
% % 训练集
% [TrainData]=readtable('FUDS_45_train.xlsx','Sheet','Sheet1');
% V_Train   = table2array(TrainData(1:1:71391,3))';  %max 71391
% A_Train   = table2array(TrainData(1:1:71391,4))';
% SOC_Train = table2array(TrainData(1:1:71391,6))';
% XTrain = [V_Train;A_Train];% XTrain = V_Train;
% YTrain = (SOC_Train);
% 
% [inputnTrain ,  inputpsTrain ] = mapminmax(XTrain);
% [outputnTrain,  outputpsTrain] = mapminmax(YTrain);
% XTrain= mapminmax('apply',XTrain,inputpsTrain);   %测试输入数据归一化
% YTrain= mapminmax('apply',YTrain,outputpsTrain);  %测试输入数据归一化

% 测试集
[TestData]=readtable('45C_FUDS_80test.xlsx','Sheet','Sheet1');
V_Test   = table2array(TestData(1:1:13519,3))';  %max 13519
A_Test   = table2array(TestData(1:1:13519,4))';
SOC_Test = table2array(TestData(1:1:13519,6))';

XTest = [V_Test;A_Test];% XTest = V_Test;
YTest = SOC_Test;
[inputnTest ,inputpsTest ] = mapminmax(XTest);
[outputnTest,outputpsTest] = mapminmax(YTest);

XTest= mapminmax('apply',XTest,inputpsTest);        %测试输入数据归一化





% BPoutput= mapminmax('reverse',YTrain,outputps);   %网络预测数据反归一化

% subplot(211);
% plot(V_Train);
% hold on;
% plot(A_Train);
% % plot(A);
% subplot(212);
% plot(SOC_Train);
% 
% %% Deep learning 深度学习框架
% % 
% % LSTM网络架构
% % 这是Bi-LSTM网络的参数，从上往下依次构建网络的输入到输出层
% layers = [ ...
%   sequenceInputLayer(2)                % 输入数据为2维数据
%   convolution1dLayer(128,2048,'Padding','same','Stride',1)
%   reluLayer('name','relu_0')               %激励函数 RELU 
%   % dropoutLayer(0.2)                         %丢弃层概率 
%   % fullyConnectedLayer(512)                  %全连接层
%   bilstmLayer(1024)                          % 
%   dropoutLayer(0.2)                         %丢弃层概率 
%   reluLayer('name','relu_1')                %激励函数 RELU 
%   fullyConnectedLayer(256)                  %全连接层
%   reluLayer('name','relu_3')                %激励函数 RELU 
%   fullyConnectedLayer(128)                  %全连接层
%   reluLayer('name','relu_4')                %激励函数 RELU 
%   fullyConnectedLayer(16)                   %全连接层
%   reluLayer('name','relu_5')                %激励函数 RELU 
%   fullyConnectedLayer(1)                    %全连接层
%   regressionLayer                           %回归层
%   ]
% 
% 
% % Bi-LSTM超参数设置
% options = trainingOptions('adam', ...   % ADAM求解器
%     'MaxEpochs',35, ...                  % 最大训练epoch次数
%     'MiniBatchSize', 256, ...           % 小批量尺寸，不宜太大，否则易出现CUDA错误
%     'InitialLearnRate', 0.001, ...       % 学习率
%     'SequenceLength', 4000, ...          % 序列长度（将信号分解成更小的片段）
%     'GradientThreshold', 1, ...         % 梯度阈值，防止梯度爆炸
%     'ExecutionEnvironment',"auto",...   % 自动选择执行的硬件环境，如果有GPU，首选GPU，否则选用CPU训练
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropPeriod',10, ...      %125次后 ，学习率下降 
%     'LearnRateDropFactor',0.2, ...      %下降因子 0.2
%     'plots','training-progress', ...    % 绘制训练过程
%     'Verbose',false);                   % 在命令行窗口展示训练过程（true:是，false:否
% 
% 
%     % 'ValidationData',{XTrain,YTrain}, ...
%     % 'ValidationFrequency',2, ...        %每1步验证一次 
% % 训练LSTM网络
% % 训练设置好的Bi-LSTM网络，并把训练好的模型存贮到对象net
% net = trainNetwork(XTrain,YTrain,layers,options);

%% 性能评估
% 步骤6：可视化训练和测试准确度
% 对训练数据进行预测
predicted_values = predict(net, XTest);     %网络预测
predicted_values = mapminmax('reverse',predicted_values,outputpsTest);   %网络预测数据反归一化

% 网络可视化
figure
subplot(221);
plot(predicted_values);
hold on
plot(YTest);
legend('predicted values', 'YTest');
xlabel('Predicted Values');
ylabel('YTest');
title('SOC plot');

subplot(222);
plot(predicted_values,YTest);
xlabel('Predicted Values');
ylabel('YTest');
title('散点图 Scatter plot');

subplot(223);
residuals = predicted_values - YTest; % 计算残差
scatter(predicted_values, residuals);   % 绘制残差图
xlabel('Predicted Values');
ylabel('Residuals');
title('残差图 Residual Plot');

subplot(224);
histogram(YTest, 'Normalization', 'pdf'); % 绘制密度图
hold on;
histogram(predicted_values, 'Normalization', 'pdf');
legend('True Values', 'Predicted Values');
xlabel('Values');
ylabel('Density');
title('密度图 Density Plot');

% 散点图：将真实值与预测值进行比较，每个点代表一个数据样本，预测值在横轴，真实值在纵轴。这有助于直观地了解预测值与真实值之间的关系。
% 
% 残差图：绘制模型的残差（预测值与真实值之间的差异）的散点图。如果模型很好，残差应该随机分布在0附近。
% 
% 密度图：显示预测值和真实值的分布情况，可以帮助你了解它们的分布是否相似。


% 计算指标评估模型的性能
true_values = YTest;    % 获取验证集中的真实目标值
error = predicted_values - true_values; % 计算预测值与真实值之间的差异
mse = mean(error .^ 2) % 计算均方误差（MSE）
mae = mean(abs(error)) % 计算平均绝对误差（MAE）
R = corrcoef(predicted_values, true_values) % 计算预测值和真实值之间的相关系数

y_true = YTest;
y_pred = predicted_values;
% 计算观测值的均值
y_mean = mean(y_true);
% 计算总平方和
SS_tot = sum((y_true - y_mean).^2);
% 计算残差
residuals = y_true - y_pred;
% 计算残差平方和
SS_res = sum(residuals.^2);
% 计算 R 方值
R_squared = 1 - (SS_res / SS_tot);
disp(['R squared value: ', num2str(R_squared)]);

figure
plot((y_pred-y_true)./y_true*100);
ylabel(['SOC Error(%)']);


%% 数据分析

% figure
% subplot(221);
% plot(predicted_values)
% 
% Y = fft(predicted_values);
% Y = Y(3:end-3);
% Y_abs = abs(Y);
% Y_max = max(abs(Y));
% subplot(222);
% plot(Y_abs/Y_max);
% 
% subplot(223);
% plot(YTest)
% 
% Y = fft(YTest);
% Y = Y(3:end-3);
% Y_abs = abs(Y);
% Y_max = max(abs(Y));
% subplot(224);
% plot(Y_abs/Y_max);
% 
% % 假设你有一组预测数据存储在变量 predictedData 中
% 
% % 创建卡尔曼滤波器对象
% kalmanFilter = vision.KalmanFilter('ProcessNoise',0.01,'MeasurementNoise',0.1);
% 
% % 初始化状态估计值为预测数据的第一个值
% kalmanFilter.State = predictedData(1);
% 
% % 对预测数据进行滤波处理
% smoothedData = zeros(size(predictedData));
% for i = 1:length(predictedData)
%     % 预测下一个状态
%     kalmanFilter.predict();
% 
%     % 校正状态估计值
%     kalmanFilter.correct(predictedData(i));
% 
%     % 获取校正后的状态估计值
%     smoothedData(i) = kalmanFilter.correctedEstimate;
% end
% 
% % 绘制结果
% figure;
% plot(predictedData, 'b.-'); % 绘制原始预测数据
% hold on;
% plot(smoothedData, 'r.-'); % 绘制平滑后的数据
% legend('Original Predicted Data', 'Smoothed Data');
% xlabel('Time');
% ylabel('Value');
% title('Kalman Filter Smoothing');








