# 模型参数
import torch
import torch.nn as nn
from torch.autograd import Variable


class GANLoss(nn.Module):
    def __init__(self,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor,
                                               requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor,
                                               requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class FocalLoss(torch.nn.Module):
    def __init__(self,
                 gamma=2,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor,
                                               requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor,
                                               requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        if target_is_real:
            ypredt = input
        else:
            ypredt = 1 - input
        gamma = 1
        eps = 1e-12
        loss = -(1.0 - ypredt)**gamma * torch.log(ypredt + eps)
        loss = loss.mean()
        return loss


def print_net(net):
    num_params = 0
    for params in net.params():
        num_params += params.numel()  # numel()返回数组中元素的总数。
    print(net)
    print("total num of parameters %d", num_params)

    # if isinstance(m, nn.Conv2d):
    #
    #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #
    #     m.weight.data.normal_(0, math.sqrt(2. / n))
    #
    # elif isinstance(m, nn.BatchNorm2d):
    #
    #     m.weight.data.fill_(1)
    #
    #     m.bias.data.zero_()
    # if isinstance(m, nn.Linear):
    #     size = m.weight.size()
    #     fan_out = size[0]  # number of rows
    #     fan_in = size[1]  # number of columns
    #     variance = np.sqrt(2.0 / (fan_in + fan_out))
    #     m.weight.data.normal_(0.0, variance)
