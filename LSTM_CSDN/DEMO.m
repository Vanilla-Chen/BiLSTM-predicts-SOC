% %% 1.环境清理
% clear, clc, close all;
% %% 2.导入数据,单序列
% D=readmatrix('B.xlsx');
% data=D(:,2);%要求行向量
% data1=data;
% % 原始数据绘图
% figure
% plot(data,'-s','Color',[0 0 255]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
% legend('原始数据','Location','NorthWest','FontName','华文宋体');
% xlabel('样本','fontsize',12,'FontName','华文宋体');
% ylabel('数值','fontsize',12,'FontName','华文宋体');
% %% 3.数据处理
% nn=1500;%训练数据集大小
% numTimeStepsTrain = floor(nn);%nn数据训练 ，N-nn个用来验证
% [XTrain,YTrain,XTest,YTest,mu,sig] = shujuchuli(data,numTimeStepsTrain);
%% 4.定义LSTM结构参数
numFeatures= 1;%输入节点
numResponses = 1;%输出节点
numHiddenUnits = 500;%隐含层神经元节点数 

%构建 LSTM网络 
layers = [sequenceInputLayer(numFeatures) 
 lstmLayer(numHiddenUnits) %lstm函数 
dropoutLayer(0.2)%丢弃层概率 
 reluLayer('name','relu')% 激励函数 RELU 
fullyConnectedLayer(numResponses)
regressionLayer]

% XTrain=XTrain';
% YTrain=YTrain';

%% 5.定义LSTM函数参数 
options = trainingOptions('adam', ... % adam优化算法 自适应学习率 
'MaxEpochs',500,...% 最大迭代次数 
 'MiniBatchSize',10, ...%最小批处理数量 
'GradientThreshold',1, ...%防止梯度爆炸 
'InitialLearnRate',0.005, ...% 初始学习率 
'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',125, ...%125次后 ，学习率下降 
'LearnRateDropFactor',0.2, ...%下降因子 0.2
'ValidationData',{XTrain,YTrain}, ...
 'ValidationFrequency',5, ...%每五步验证一次 
'Verbose',1, ...
 'Plots','training-progress');


%% 6.训练LSTM网络 
net = trainNetwork(XTrain,YTrain,layers,options);

% %% 7.建立训练模型 
% net = predictAndUpdateState(net,XTrain);
% 
% %% 8.仿真预测(训练集) 
% M = numel(XTrain);
% for i = 1:M
%     [net,YPred_1(:,i)] = predictAndUpdateState(net,XTrain(:,i),'ExecutionEnvironment','cpu');%
% end
% T_sim1 = sig*YPred_1 + mu;%预测结果去标准化 ，恢复原来的数量级 
% %% 9.仿真预测(验证集) 
% N = numel(XTest);
% for i = 1:N
%     [net,YPred_2(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');%
% end
% T_sim2 = sig*YPred_2 + mu;%预测结果去标准化 ，恢复原来的数量级 
% %% 10.评价指标
% %  均方根误差
% T_train=data1(1:M)';
% T_test=data1(M+1:end)';
% error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
% error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
% %  MAE
% mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
% disp(['训练集数据的MAE为：', num2str(mae1)])
% disp(['验证集数据的MAE为：', num2str(mae2)])
% %  MAPE
% maep1 = sum(abs(T_sim1 - T_train)./T_train) ./ M ;
% maep2 = sum(abs(T_sim2 - T_test )./T_test) ./ N ;
% disp(['训练集数据的MAPE为：', num2str(maep1)])
% disp(['验证集数据的MAPE为：', num2str(maep2)])
% %  RMSE
% RMSE1 = sqrt(sumsqr(T_sim1 - T_train)/M);
% RMSE2 = sqrt(sumsqr(T_sim2 - T_test)/N);
% disp(['训练集数据的RMSE为：', num2str(RMSE1)])
% disp(['验证集数据的RMSE为：', num2str(RMSE2)])
% %% 11. 绘图
% figure
% subplot(2,1,1)
% plot(T_sim1,'-s','Color',[255 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[250 0 0]./255)
% hold on 
% plot(T_train,'-o','Color',[150 150 150]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[150 150 150]./255)
% legend( 'LSTM拟合训练数据','实际分析数据','Location','best');
% title('LSTM模型预测结果及真实值','fontsize',12)
% xlabel('样本','fontsize',12);
% ylabel('数值','fontsize',12);
% xlim([1 M])
% %-------------------------------------------------------------------------------------
% subplot(2,1,2)
% bar((T_sim1 - T_train)./T_train)   
% legend('LSTM模型训练集相对误差','Location','best')
% title('LSTM模型训练集相对误差','fontsize',12)
% ylabel('误差','fontsize',12)
% xlabel('样本','fontsize',12)
% xlim([1 M]);
% %-------------------------------------------------------------------------------------
% figure
% subplot(2,1,1)
% plot(T_sim2,'-s','Color',[0 0 255]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
% hold on 
% plot(T_test,'-o','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[0 0 0]./255)
% legend('LSTM预测测试数据','实际分析数据','Location','best');
% title('LSTM模型预测结果及真实值','fontsize',12)
% xlabel('样本','fontsize',12);
% ylabel('数值','fontsize',12);
% xlim([1 N])
% %-------------------------------------------------------------------------------------
% subplot(2,1,2)
% bar((T_sim2 - T_test )./T_test)   
% legend('LSTM模型测试集相对误差','Location','NorthEast')
% title('LSTM模型测试集相对误差','fontsize',12)
% ylabel('误差','fontsize',12)
% xlabel('样本','fontsize',12)
% xlim([1 N]);
% 
% %% 12.预测未来
% P = N-nn;% 预测未来数量
% YPred_3 = [];%预测结果清零 
% [T_sim3] = yuceweilai(net,XTrain,data,P,YPred_3,sig,mu)
% 
% %%  13.绘图
% figure
% plot(1:size(data,1),data,'-s','Color',[255 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[250 0 0]./255)
% hold on 
% %plot(size(data,1)+1:size(data,1)+P,T_sim3,'-o','Color',[150 150 150]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[150 150 150]./255)
% legend( 'LSTM预测结果','Location','NorthWest');
% title('LSTM模型预测结果','fontsize',12)
% xlabel('样本','fontsize',12);
% ylabel('数值','fontsize',12);


%% 
function [XTrain,YTrain,XTest,YTest,mu,sig] = shujuchuli(data,numTimeStepsTrain)
dataTrain = data(1:numTimeStepsTrain+1,:);% 训练样本
dataTest = data(numTimeStepsTrain:end,:); %验证样本 
%训练数据标准化处理 
mu = mean(dataTrain,'ALL');
sig = std(dataTrain,0,'ALL');
dataTrainStandardized = (dataTrain - mu) / sig;
XTrain = dataTrainStandardized(1:end-1,:);% 训练输入 
YTrain = dataTrainStandardized(2:end,:);% 训练输出
%测试样本标准化处理 
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1,:);%测试输入 
YTest = dataTest(2:end,:);%测试输出 
 
XTest=XTest';
YTest=YTest';
end