"""
script is used to test a U-Net model based on a previously trained model. Script computes IoU and performance metrics used in paper.

"""
import os
import glob
from img_segmentation.model import UNet


# for windows
tune_vid = ''
# path to project root
file_base = 'E:\\phd_data\\2017_watsen'  #
#
model_names = ['l5f16res_finetuned_', 'l5f16res_finetuned_FloodXCam5', 'l5f16res_finetuned_HoustonGarage']
camera_names = ['FloodXCam1', 'FloodXCam5', 'HoustonGarage']
# use additional augmentation
aug = [False, False, False]
feat = [16, 16, 16]
ep = [100, 100, 100]
lay = [5, 5, 5]
drop = [0.75, 0.75, 0.75]
bat = [1, 1, 1]
res = [True, True, True]
# base network on previously trained model
bd = [os.path.join(file_base, 'models', 'l5f16res_augmented'), os.path.join(file_base, 'models', 'l5f16res_augmented'), os.path.join(file_base, 'models', 'l5f16res_augmented')]

rois = {
    'AthleticPark': [102, 171, 327, 236],
    'FloodXCam1': [275, 136, 174, 62],
    'FloodXCam5': [8, 239, 101, 43],
    'HoustonGarage': [185, 114, 205, 217],
    'HarveyParking': [127, 267, 95, 105],
    'BayouBridge': [58, 124, 401, 259]
}

img_shape = (512, 512, 3)

test_dir = os.path.join(file_base, 'cctv_masks', '*')

# with roi
for i, model_name in enumerate(model_names):
    if i in [0]:

        unet = UNet(img_shape, root_features=feat[i], layers=lay[i], batch_norm=True, dropout=drop[i], inc_rate=2.,
                    residual=res[i])

        for test in glob.glob(test_dir):  # test for all frames in directory

            base, tail = os.path.split(test)

            model_dir = os.path.join(file_base, 'models', model_name + tail)
            pred = os.path.join(model_dir, 'pred_roi_' + tail)
            csv_path = os.path.join(model_dir, tail + '_roi.csv')
            test_val = os.path.join(test, 'validate')
            if not os.path.isdir(pred):
                os.mkdir(pred)

            unet.test(model_dir, [test_val], pred, csv_path=csv_path, roi=rois[tail])

# without roi
for i, model_name in enumerate(model_names):
    if i in [0]:

        unet = UNet(img_shape, root_features=feat[i], layers=lay[i], batch_norm=True, dropout=drop[i], inc_rate=2.,
                    residual=res[i])

        for test in glob.glob(test_dir):  # test for all frames in directory
            base, tail = os.path.split(test)

            model_dir = os.path.join(file_base, 'models', model_name + tail)
            pred = os.path.join(model_dir, 'pred_roi_' + tail)
            csv_path = os.path.join(model_dir, tail + '.csv')
            test_val = os.path.join(test, 'validate')
            if not os.path.isdir(pred):
                os.mkdir(pred)

            unet.test(model_dir, [test_val], pred, csv_path=csv_path, roi=None)