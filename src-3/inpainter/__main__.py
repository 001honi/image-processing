import argparse
from skimage.io import imread, imsave

from inpainter import Inpainter


def main():
    args = parse_args()

    image = imread(args.input_image)
    mask = imread(args.mask, as_gray=True)

    # args.output = f'out-{args.patch_size}-{args.patch_num}-{args.lambda_dist}.jpg'
    print(args)

    output_image = Inpainter(
        image,
        mask,
        patch_size=args.patch_size,
        plot_progress=args.plot_progress,
        k=args.patch_num,
        lambda_dist=args.lambda_dist
    ).inpaint()
    imsave(args.output, output_image, quality=100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ps',
        '--patch-size',
        help='the size of the patches',
        type=int,
        default=5
    )
    parser.add_argument(
        '-pn',
        '--patch-num',
        help='the number of patches combined in source region',
        type=int,
        default=1
    )
    parser.add_argument(
        '-dist',
        '--lambda-dist',
        help='regularization for euclidean distance',
        type=int,
        default=1
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the file path to save the output image',
        default='output.jpg'
    )
    parser.add_argument(
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--input_image',
        help='the image containing objects to be removed',
        default='inputs/image5.png'
    )
    parser.add_argument(
        '--mask',
        help='the mask of the region to be removed',
        default= 'inputs/mask5.png'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
