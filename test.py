# 外部传参
from option import args
# 调用函数
from data_loader import MyDataset
from networks import Generator
# 模型参数
import os
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    net_G = Generator(args.input_nc, args.output_nc)
    net_G.load_state_dict(torch.load(args.checkpoints + '/net_G_ins.pth'))
    net_G.cuda()
    print("net_G loaded")

    dataloader = MyDataset(args, isTrain=1)
    imgNum = len(dataloader)
    print(len(dataloader))

    test_loader = torch.utils.data.DataLoader(dataset=dataloader,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=2)

    fakeB = torch.FloatTensor(imgNum, args.output_nc, args.img_height,
                              args.img_width)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for i, image in enumerate(test_loader):
        imgA = image[0]

        real_A = Variable(imgA.cuda())
        fake_B = net_G(real_A)

        fake_B_unpad = unpad(fake_B.data, args)

        fakeB[i, :, :, :] = fake_B_unpad

        print("%d.jpg generate completed" % i)

        vutils.save_image(fake_B_unpad,
                          '%s/fakeB_%s.png' % (args.output, str(i)),
                          normalize=True,
                          scale_each=True)

    vutils.save_image(fakeB,
                      '%s/fakeB_concat.png' % (args.output),
                      normalize=True,
                      scale_each=True,
                      padding=2)


def unpad(img, args):
    return img[:, :, args.fine_pad_h:(
        -args.fine_pad_h if args.fine_pad_h != 0 else None), args.fine_pad_w:(
            -args.fine_pad_w if args.fine_pad_w != 0 else None)]


if __name__ == '__main__':
    main()