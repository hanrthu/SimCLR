import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


# parameters
valid_split = 0.1
batch_size = 64
T1 = 10
T2 = 100
lr1 = 0.001
lr2 = 0.001
l1 = 0.0
l2 = 0.0
# weight_decay = 0.0001


# data
train_data = numpy.load('course_train.npy')
train_inputs = torch.Tensor(train_data[:, :-2])
train_labels = torch.Tensor(train_data[:, -2:-1]).long().squeeze()
train_set = Data.TensorDataset(train_inputs, train_labels)
valid_size = int(valid_split * len(train_data))
train_size = int(len(train_data) - valid_size)
train_set, valid_set = Data.random_split(train_set, [train_size, valid_size])
train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = Data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)


# model
class CNBB(nn.Module):
    def __init__(self):
        super(CNBB, self).__init__()
        self.g = nn.Sequential(
            nn.Linear(512, 50),
            nn.Tanh()
        )
        self.f = nn.Sequential(
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )
        self.w = nn.Sequential(
            nn.Linear(50, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        features = self.g(inputs)
        weights = self.w(features)
        outputs = self.f(features)
        return features, weights, outputs


def get_lossb(X, W, I):
    M1 = torch.mm(X.t(), W * I)
    M2 = torch.mm(X.t(), W * (1.0 - I))
    M1 -= torch.diagflat(torch.diagonal(M1))    # -j
    M2 -= torch.diagflat(torch.diagonal(M2))
    v1 = 1.0 / torch.mm(W.t(), I)
    v2 = 1.0 / torch.mm(W.t(), 1.0 - I)
    v1[v1 == float('inf')] = 0.0
    v2[v2 == float('inf')] = 0.0
    return torch.sum((M1 * v1 - M2 * v2) ** 2.0)


def train(net, train_loader, valid_loader):
    optim1 = torch.optim.Adam([{'params': net.g.parameters()}, 
                               {'params': net.f.parameters()}], lr=lr1)
    optim2 = torch.optim.Adam(net.w.parameters(), lr=lr2)
    
    net.train()
    for t1 in range(T1):
        # Sample batch of images      
        for index, data in enumerate(train_loader):   
            inputs, labels = data

            # Extract image features
            features = net.g(inputs)
            # print(features)

            # Calculate indicator matrix I of features
            indicators = (features > 0).float()
            # print(indicators)
            # for j in range(indicators.size()[1]):
            #     print(indicators[:, j])
            #     if torch.equal(indicators[:, j], torch.ones(indicators.size()[0])):
            #         print('I_%d = torch.ones' % j)
            #     if torch.equal(indicators[:, j], torch.zeros(indicators.size()[0])):
            #         print('I_%d = torch.zeros' % j)

            # Optimize f to minimize Lossb 
            for t2 in range(T2):
                optim2.zero_grad()
                weights = net.w(features)
                print(weights.size())
                loss_b = l2 * torch.sum(weights ** 2.0) # regulation
                # p = features.size()[1]
                # for j in range(p):
                #     trans = torch.ones(p)
                #     trans[j] = 0.0
                #     if torch.equal(indicators[:, j], torch.ones(indicators.size()[0]))  \
                #     or torch.equal(indicators[:, j], torch.zeros(indicators.size()[0])):
                #         continue
                #     cons = torch.mm((features * trans).t(), weights * indicators[:, [j]]) / torch.mm(weights.t(), indicators[:, [j]])    \
                #          - torch.mm((features * trans).t(), weights * (1.0 - indicators[:, [j]])) / torch.mm(weights.t(), (1.0 - indicators[:, [j]]))
                #     loss_b += torch.sum(cons ** 2.0)
                # print(loss_b.item())
                # loss_b.backward(retain_graph=True)
                loss_b += get_lossb(features, weights, indicators)
                loss_b.backward()
                optim2.step()
                print(loss_b.item())
                input()
            input()
            #     print(weights)

            # Optimize g and f to minimize Lossp
            optim1.zero_grad()
            outputs = net.f(features)
            loss_q = - torch.sum(outputs ** 2.0)
            loss_p = l1 * loss_q   # regulation
            loss_p += torch.sum(weights * torch.log(outputs[torch.arange(len(outputs)), labels]))
            loss_p.backward()
            optim1.step()
            # print(loss_p)
            # print(outputs)

            print('epoch: %2d batch: %2d' % (t1, index))
    return


# train & test
net = CNBB()
train(net, train_loader, valid_loader)

# summary(net, (512,), batch_size=16, device='cpu')
# for name in net.state_dict().keys():
    # print(name)
    # print(net.state_dict()[name])