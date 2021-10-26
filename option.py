import argparse

parser = argparse.ArgumentParser(description="Pix2Pix")

parser.add_argument('--dataroot', required=True, help="path to images")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')

parser.add_argument('--img_height', default=250, type=int)  # 数据集中的图像高度
parser.add_argument('--img_width', default=200, type=int)  # 数据集中的图像宽度
parser.add_argument('--load_size', type=int, default=286)
parser.add_argument('--fine_size', default=256, type=int)  # 神经网络中的图像大小

parser.add_argument('--input_nc', type=int, default=3)
parser.add_argument('--output_nc', type=int, default=3)

parser.add_argument('--lr', type=int, default=1e-4, help='learning rate')
parser.add_argument('--bata',
                    type=int,
                    default=0.5,
                    help='momentum parameters bata1')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--niter',
                    type=int,
                    default=600,
                    help='number of epochs to train for')
parser.add_argument('--lamb',
                    type=int,
                    default=100,
                    help='weight on L1 term in objective')
parser.add_argument('--sample',
                    type=str,
                    default='./sample',
                    help='models are saved here')
parser.add_argument('--checkpoints',
                    type=str,
                    default='./checkpoints',
                    help='image are saved here')
parser.add_argument('--output',
                    default='./output',
                    help='folder to output images ')
parser.add_argument('--datalist', default='files/list_train.txt')
parser.add_argument(
    '--pre_net', default='/home/kejia/PycharmProjects/pix2pix_xxx/experiment/')

args = parser.parse_args()

# 错误检查
assert (args.load_size >= args.img_height and args.load_size >= args.img_width
        ), "img_size must be bigger than origin_height and origin_width"

args.load_pad_h = (args.load_size - args.img_height) // 2
args.load_pad_w = (args.load_size - args.img_width) // 2

args.fine_pad_h = (args.fine_size - args.img_height) // 2
args.fine_pad_w = (args.fine_size - args.img_width) // 2

args.fill_h = args.load_size - args.fine_size
args.fill_w = args.load_size - args.fine_size