%%% 载入数据集并对种类进行标签化
load fisheriris
label = zeros(150,1);
label(strcmp(species,'versicolor')) = 1;
label(strcmp(species,'virginica')) = 2;

%{%%%可视化并分析数据
%hf1=figure;
    %绘制散点图矩阵
%speciesNumer = grp2idx(species);
%[H,AX,BigAx] = gplotmatrix(meas,[],speciesNumer,['r' 'g' 'b']);
% legend(AX(13+3),{'山鸢尾','多色鸢尾','弗吉尼亚鸢尾'},'FontWeight','Bold','Fontsize',10)
% title(BigAx,'鸢尾花数据特征散点图','FontWeight','Bold','Fontsize',16)
    %横坐标
% xlabel(AX(4),{'萼片长度'},'FontWeight','Bold','Fontsize',12)
% xlabel(AX(8+1),{'萼片宽度'},'FontWeight','Bold','Fontsize',12)
% xlabel(AX(12+2),{'花瓣长度'},'FontWeight','Bold','Fontsize',12)
% xlabel(AX(16+3),{'花瓣宽度'},'FontWeight','Bold','Fontsize',12)
    % 纵坐标
% ylabel(AX(1),{'萼片长度'},'FontWeight','Bold','Fontsize',12)
% ylabel(AX(2),{'萼片宽度'},'FontWeight','Bold','Fontsize',12)
% ylabel(AX(3),{'花瓣长度'},'FontWeight','Bold','Fontsize',12)
% ylabel(AX(4),{'花瓣宽度'},'FontWeight','Bold','Fontsize',12)
%}
%%%随机化并取训练集和测试集
%n = randperm(size(meas,1);
%y = label(n);
%train_features = meas(n(1:120),[1:2]);
%train_labels = label(n(1:120),:);
lam= 0;
meas = meas(:,(1:2));
test_features = [meas(1:10,:);meas(91:100,:);meas(141:150,:)];
test_labels = [label(1:10,:);label(91:100,:);label(141:150,:)];

x_train1 = meas(11:90,:);
x_train2 = meas(61:140,:);
x_train3 = [meas(11:50,:);meas(101:140,:)];
y_train1 = label(11:90,:);
y_train2 = label(61:140,:)-1;
y_train3 = [label(11:50,:);label(101:140,:)]-1;

%%%进行预测
theta1 = logisticRegression(x_train1,y_train1,lam);
theta2 = logisticRegression(x_train2,y_train2,lam);
theta3 = logisticRegression(x_train3,y_train3,lam);
X_b = [test_features ,ones(size(test_features,1),1)];
y1 = pro(X_b,theta1);
y2 = pro(X_b,theta2);
y3 = pro(X_b,theta3);
y_predict = zeros(size(test_labels,1),1);
accuracy = 0;
for i = 1:size(y_predict,1)
    if ( y1(i) )
        if ( y2(i))
            y_predict(i) = 2;
        else
            y_predict(i) = 1;
        end
    else
        if ( y2(i) )
            if ( y3(i) )
                y_predict(i) = 2;
            else
                y_predict(i) = 0;
            end
        else           
            y_predict(i) = 0;
        end
    end
    if (y_predict(i) == test_labels(i))
        accuracy = accuracy + 1;
    end
end
accuracy = accuracy./size(test_labels,1);
fprintf('lamda = %f\n',lam);
fprintf('准确度为 %f%%', 100*accuracy);

hf2 = figure;
hold on;
plot(y_predict,'o');
plot(test_labels,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('预测测试集分类','实际测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
 grid on;

function y = sigmoid(z)
    y = 1./(1 + exp(-z));
end
function J = lossfunction(theta,X_b,y,lamda)
    p_predict = sigmoid(X_b * theta);
    punish = (lamda./(2*size(y,1)))*sum(theta.^2);
    J = - sum(y.*log(p_predict+0.001) + (1 - y).*log(1-p_predict+0.001))./size(y,1)+punish;
end
function dJ = deri(theta,X_b,y)
    p_predict = sigmoid(X_b * theta);
    dJ = ((X_b.') * (p_predict - y))./size(X_b,1);
end
function theta = g_decent(X_b,y,theta0,eta,lamda)
    N = 10000;
    epsilon = 1e-8;
    theta = theta0;
    i = 0;
    while i < N
        grad = deri(theta,X_b,y);
        hold_theta = theta;
        i = i + 1;
        theta = theta - grad .* eta;
        for I = 2:size(theta,1)
            theta = theta - grad .* eta - (lamda*theta(I)./size(y,1));
        end
        
        if(abs(lossfunction(theta,X_b,y,lamda) - lossfunction(hold_theta,X_b,y,lamda)) < epsilon)
            break;
        end
    end
end
function theta = logisticRegression(x_train ,y,lam)
    X_b = [x_train , ones(size(x_train,1),1)];
    theta0 = normrnd(0,1,[size(X_b,2),1]);
    theta = g_decent(X_b,y,theta0,0.01,lam);
end
function bool = pro(X_b,theta)
    predict = sigmoid(X_b * theta);
    predict(predict >= 0.5) = 1;
    predict(predict < 0.5) = 0;
    bool = predict;
end