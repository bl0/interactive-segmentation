import argparse
import os
import os.path as osp
from pprint import pprint
import textwrap

import cv2
import numpy as np

from utils import COLOR_BG, COLOR_FG
from utils import load_image, load_mask, mask2color
from utils import linear_combine, bool_str, CustomFormatter


draw_color = 1
cur_mouse = (-1, -1)
left_mouse_down = False
mask = np.zeros([])
show_img = np.zeros([])
radius = 3
print('definition', id(show_img))


def on_mouse(event, x, y, flags, _):
    global left_mouse_down, cur_mouse, radius
    cur_mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        left_mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        left_mouse_down = False
    if left_mouse_down and mask.size > 0 and show_img.size > 0:
        print('on_mouse', id(show_img))
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            cv2.circle(show_img, (x, y), radius, COLOR_BG, -1)
            cv2.circle(mask, (x, y), radius, cv2.GC_BGD, -1)
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.circle(show_img, (x, y), radius, COLOR_FG, -1)
            cv2.circle(mask, (x, y), radius, cv2.GC_FGD, -1)


def main(args):
    files = sorted([x for x in os.listdir(args.img_dir)
                    if '.png' in x or '.jpg' in x.lower()])

    # jump over processed images
    idx = 0
    # while idx < len(files) and osp.exists(osp.join(args.save_dir, files[idx])):
    #     idx += 1

    cv2.namedWindow(args.windows_name)
    cv2.setMouseCallback(args.windows_name, on_mouse)
    cv2.createTrackbar('brush size', args.windows_name,
                       3, args.max_radius, lambda x: None)

    global draw_color, mask, show_img, radius
    print('after gloabl', id(show_img))
    while idx < len(files):
        img_path = osp.join(args.img_dir, files[idx])
        mask_path = osp.join(args.save_dir, files[idx])
        print('process %s' % files[idx])

        img = load_image(img_path, args.max_height, args.max_width)
        mask = load_mask(mask_path, img.shape, args.use_prev_mask)
        mask[mask == 0] = 2  # convert GC_BGD to GC_PR_BGD
        # mask[mask == 1] = 3  # convert GC_FGD to GC_PR_FGD
        print('after load mask', id(show_img))

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        while True:
            radius = cv2.getTrackbarPos('brush size', args.windows_name)
            show_img = linear_combine(img, mask2color(mask), [0, 0.7, 1][draw_color]).astype('uint8')

            cv2.circle(show_img, cur_mouse, radius,
                       (200, 200, 200), (2 if left_mouse_down else 1))
            cv2.imshow(args.windows_name, show_img)
            key = cv2.waitKey(100)

            if key == ord('w'):
                draw_color = (draw_color + 1) % 3
            elif key == ord('e'):
                draw_color = (draw_color - 1) % 3
            elif key == 32:  # space
                print('segmenting...', end='')
                cv2.waitKey(1)
                # mask enum
                # GC_BGD    = 0,  //背景
                # GC_FGD    = 1,  //前景
                # GC_PR_BGD = 2,  //可能背景
                # GC_PR_FGD = 3   //可能前景
                hist, _ = np.histogram(mask, [0, 1, 2, 3, 4])
                if hist[0] + hist[2] != 0 and hist[1] + hist[3] != 0:
                    print('grabcut: ', id(show_img))
                    cv2.grabCut(img, mask, None, bgd_model, fgd_model,
                                args.iter_count, cv2.GC_INIT_WITH_MASK)
                print('done')
            elif key == ord('s') or key == 10:
                cv2.imwrite(mask_path, mask2color(mask))
                print('save label %s.' % mask_path)
                idx += 1
                break
            elif key == ord('p') and idx > 0:
                idx -= 1
                break
            elif key == ord('n') or key == 32:
                idx += 1
                break
            elif key == ord('q') or key == 27:
                return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=CustomFormatter,
        description=textwrap.dedent("""GUI usage:
            CTRL+left mouse button: select certain background pixels
            left mouse button: select certain foreground pixels
            SPACE: run segmentation again
            'p': prev image \t 'n': next image
            's'/ENTER: save label \t 'q'/ESC: exit
            'w'/'e': change alpha"""))
    parser.add_argument('-i', '--img-dir', type=str, default='data/images',
                        metavar='DIR', help='directory contains input images')
    parser.add_argument('-o', '--save-dir', type=str, default='data/annotation',
                        metavar='DIR', help='directory for segmentation results')
    parser.add_argument('--windows-name', type=str,
                        default='Interactive Segmentation', help='title of main window')
    parser.add_argument('--max-radius', type=int,
                        default=40, help='max radius of brush')
    parser.add_argument('--max-height', type=int,
                        default=400, help='max height of image')
    parser.add_argument('--max-width', type=int,
                        default=600, help='max width of image')
    parser.add_argument('--use-prev-mask', type=bool_str,
                        default=True, help='if use previous mask')
    parser.add_argument('--iter-count', type=int, default=1,
                        help='#iter parameter for grabCut algorithms')
    cfg = parser.parse_args()
    pprint(cfg)

    if not osp.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
        print('%s not exists, create it.' % cfg.save_dir)

    main(cfg)
