"""
script is used to fine-tune a U-Net model based on a previously trained model.

"""
import os
from img_segmentation.model import UNet


# for windows
tune_vid = ''
# path to project root
file_base = 'E:\\phd_data\\2017_watsen'  #
#
model_names = ['l5f16res_finetuned_FloodXCam1', 'l5f16res_finetuned_FloodXCam5', 'l5f16res_finetuned_HoustonGarage']
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

for i, model_name in enumerate(model_names):
    if i in [2]:
        model_dir = os.path.join(file_base, 'models', model_name)

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        # paths for fine-tuning
        train_dir = os.path.join(file_base, 'cctv_masks', camera_names[i], 'train')
        valid_dir = os.path.join(file_base, 'cctv_masks', camera_names[i], 'validate')

        pred_dir_flood = os.path.join(file_base, 'models', model_name, 'test_img')
        if not os.path.isdir(pred_dir_flood):
            os.mkdir(pred_dir_flood)

        pred_dir = os.path.join(file_base, 'predictions', model_name)
        if not os.path.isdir(pred_dir):
            os.mkdir(pred_dir)

        test_dir = os.path.join(file_base, 'video_masks', '*')

        img_shape = (512, 512, 3)
        unet = UNet(img_shape, root_features=feat[i], layers=lay[i], batch_norm=True, dropout=drop[i], inc_rate=2., residual=res[i])

        # retrain model, using augmented network as basis and only training last two layers (index 14)
        unet.train(model_dir, [train_dir], [valid_dir],
                   batch_size=bat[i], epochs=ep[i],
                   augmentation=aug[i], base_dir=bd[i],
                   save_aug=False, learning_rate=0.001,
                   trainable_index=14
                   )
        # now retrain model, using previous network as basis and retraining all weights
        # model_dir_fine = os.path.join(file_base, 'models', model_name + '_final')
        unet.train(model_dir=model_dir, train_dir=[train_dir], valid_dir=[valid_dir],
                   batch_size=bat[i], epochs=ep[i],
                   augmentation=aug[i], base_dir=model_dir,
                   save_aug=False, learning_rate=0.001,
                   trainable_index='all'
                   )
