from sofi_extraction.engine import CCTVFloodExtraction
import os


file_base = 'E:\\phd_data\\2017_watsen'

video_file = os.path.join(file_base, 'videos', 'floodX_cam1.mp4')

model_name = 'l5f16res_augmented'  # 'ft_l5b3e200f16_dr075i2res_lr'  # 'ft_l5b3e200f16_dr075i2res_lr'

frames = {
    'name': ['FloodXCam1',
             'FloodXCam5'],
    # 'name': ['FloodXCam1_exp20-22',
    #          'FloodXCam5_exp20-22'],
    'roi': [
        [275, 136, 174, 62],
        [199, 185, 126, 86]],
    'fps': [1, 1],
    'ref': ['file_name',
            'file_name'],
    # 'model': ['l5f16res_augmented',
    #           'l5f16res_augmented']
    'model': ['l5f16res_finetuned_FloodXCam1',
              'l5f16res_augmented']
}

if not os.path.exists(file_base):
    os.mkdir(file_base)


for i, name in enumerate(frames['name']):
    if i==1:
        print('processing {}'.format(name))
        pred_dir_flood = os.path.join(file_base, 'predictions', frames['model'][i])
        frame_dir_flood = os.path.join(file_base, 'frames')
        vid_dir_flood = os.path.join(pred_dir_flood, name + '_pred.avi')
        signal_dir_flood = os.path.join(pred_dir_flood, 'predictions', frames['model'][i])
        ref_path = frames['ref'][i]
        model_file = os.path.join(file_base, 'models', frames['model'][i])
        cr_win = dict(left=frames['roi'][i][0], top=frames['roi'][i][1], width=frames['roi'][i][2], height=frames['roi'][i][3])
        cfe = CCTVFloodExtraction(video_file, model_file, pred_dir=pred_dir_flood, frame_dir=frame_dir_flood,
                                  video_name=name, crop_window=cr_win)
        cfe.run(['extract_trend'], ref_path=ref_path)
