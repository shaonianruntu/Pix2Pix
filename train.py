# 外部传参
from option import args
# 调用函数
from data_loader import MyDataset
from networks import Generator, Discriminator, weights_init
from loss import GANLoss
# 模型参数
import os
import time
import torch
import torch.nn as nn
import torchvision.utils as vutils

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    data_loader = MyDataset(args)
    dataset_size = len(data_loader)
    print('trainA images = %d' % dataset_size)

    train_loader = torch.utils.data.DataLoader(dataset=data_loader,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2)

    net_G = Generator(args.input_nc, args.output_nc)
    net_D = Discriminator(args.input_nc, args.output_nc)
    net_G.cuda()
    net_D.cuda()
    net_G.apply(weights_init)
    net_D.apply(weights_init)
    print(net_G)
    print(net_D)

    criterionGAN = GANLoss()
    critertionL1 = nn.L1Loss()

    optimizerG = torch.optim.Adam(net_G.parameters(),
                                  lr=args.lr,
                                  betas=(args.bata, 0.999))
    optimizerD = torch.optim.Adam(net_D.parameters(),
                                  lr=args.lr,
                                  betas=(args.bata, 0.999))

    net_D.train()
    net_G.train()

    if not os.path.exists(args.sample):
        os.makedirs(args.sample)

    for epoch in range(1, args.niter + 1):

        epoch_start_time = time.time()

        for i, image in enumerate(train_loader):

            imgA = image[0]
            imgB = image[1]

            real_A = imgA.cuda()
            real_B = imgB.cuda()

            fake_B = net_G(real_A)

            net_D.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = net_D(fake_AB.detach())

            loss_D_fake = criterionGAN(pred_fake, False)

            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = net_D(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()

            optimizerD.step()

            # netG
            net_G.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            out_put = net_D(fake_AB)
            loss_G_GAN = criterionGAN(out_put, True)

            loss_G_L1 = critertionL1(fake_B, real_B) * args.lamb
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(
                    '[%d/%d][%d/%d] LOSS_D: %.4f LOSS_G: %.4f LOSS_L1: %.4f' %
                    (epoch, args.niter, i, len(train_loader), loss_D, loss_G,
                     loss_G_L1))
                print('LOSS_real: %.4f LOSS_fake: %.4f' %
                      (loss_D_real, loss_D_fake))

        print('Time Taken: %d sec' % (time.time() - epoch_start_time))

        if epoch % 5 == 0:
            vutils.save_image(fake_B.data,
                              args.sample + '/fake_samples_epoch_%03d.png' %
                              (epoch),
                              normalize=True)

        if epoch >= 500:
            if not os.path.exists(args.checkpoints):
                os.makedirs(args.checkpoints)
            if epoch % 100 == 0:
                torch.save(
                    net_G.state_dict(),
                    args.checkpoints + '/net_G_ins' + str(epoch) + '.pth')
                torch.save(
                    net_D.state_dict(),
                    args.checkpoints + '/net_D_ins' + str(epoch) + '.pth')
                print("saved model at epoch " + str(epoch))

    print("save net")
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    torch.save(net_G.state_dict(), args.checkpoints + '/net_G_ins.pth')
    torch.save(net_D.state_dict(), args.checkpoints + '/net_D_ins.pth')


if __name__ == '__main__':
    main()
