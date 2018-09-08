import numpy as np


def mask(im_sh=(64, 3, 64, 64), crop=4, max_crop=16, frac=0.1, max_frac=0.1, maskType='rand_rect'):
    msk = np.ones(im_sh)
    if maskType == 'rand_rect':
        pos_h = int(np.random.randint(im_sh[2] - crop + 1,  size=1))
        pos_w = int(np.random.randint(im_sh[3] - crop + 1,  size=1))
        msk[:, :, pos_h:pos_h + crop, pos_w:pos_w + crop] = 0.0
    elif maskType == 'rand_rect_crop':
        crop = int(np.random.randint(max_crop,  size=1))
        pos_h = int(np.random.randint(im_sh[2] - crop + 1,  size=1))
        pos_w = int(np.random.randint(im_sh[3] - crop + 1,  size=1))
        msk[:, :, pos_h:pos_h + crop, pos_w:pos_w + crop] = 0.0
    elif maskType == 'rand_pix_frac':
        frac = np.random.randint(int(max_frac * 1000), size=1) / 1000.
        thres = np.random.random((1, 1, im_sh[2], im_sh[3]))
        thres = np.tile(thres, (im_sh[0], im_sh[1], 1, 1))
        msk[thres < frac] = 0.0
    else:
        thres = np.random.random((1, 1, im_sh[2], im_sh[3]))
        thres = np.tile(thres, (im_sh[0], im_sh[1], 1, 1))
        msk[thres < frac] = 0.0
    sgn = np.random.randint(2,  size=1) * 2 - 1
    bdy = (1 - msk) * sgn
    return msk.astype(int), bdy.astype(int)


