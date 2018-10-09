import os
from sofi_extraction.engine import CCTVFloodExtraction

# for windows
file_base = 'E:\\phd_data\\2017_watsen'

model_basic = 'l5f16res_basic'
model_augmented = 'l5f16res_augmented'


frames = {
    'name': ['AthleticPark',
             'FloodXCam1',
             'FloodXCam5',
             'HoustonGarage',
             'HarveyParking',
             'BayouBridge'],
    'roi': [
        [102, 171, 327, 236],
        [275, 136, 174, 62],
        [8, 239, 101, 43],
        [185, 114, 205, 217],
        [127, 267, 95, 105],
        [58, 124, 401, 259]],
    'fps': [1, 15, 15, 15, 15, 1],
    'ref': [os.path.join(file_base, 'cctv_footage', 'AthleticPark', 'level_AthleticPark.csv'),
            'file_name',
            'file_name',
            os.path.join(file_base, 'cctv_footage', 'HoustonGarage', 'level_HoustonGarage.csv'),
            os.path.join(file_base, 'cctv_footage', 'HarveyParking', 'level_HarveyParking.csv'),
            os.path.join(file_base, 'cctv_footage', 'BayouBridge', 'level_BayouBridge.csv')],
    'model': ['l5f16res_finetuned_AthleticPark',
              'l5f16res_finetuned_FloodXCam1',
              'l5f16res_finetuned_FloodXCam5',
              'l5f16res_finetuned_HoustonGarage',
              'l5f16res_finetuned_HarveyParking',
              'l5f16res_finetuned_BayouBridge']
}

for i, name in enumerate(frames['name']):
    if i in [0, 5]:
        for model_name in [model_augmented, model_basic, frames['model'][i]]:
            print('processing {} with model {}'.format(name, model_name))
            pred_dir_flood = os.path.join(file_base, 'predictions', frames['model'][i])
            frame_dir_flood = os.path.join(file_base, 'frames')
            vid_dir_flood = os.path.join(pred_dir_flood, name + '_pred.avi')
            signal_dir_flood = os.path.join(pred_dir_flood, 'predictions', frames['model'][i])
            ref_path = frames['ref'][i]
            video_file = os.path.join(file_base, 'cctv_footage', '{}.mp4'.format(frames['name'][i]))
            model_file = os.path.join(file_base, 'models', model_name)
            cr_win = dict(left=frames['roi'][i][0], top=frames['roi'][i][1], width=frames['roi'][i][2], height=frames['roi'][i][3])
            cfe = CCTVFloodExtraction(video_file, model_file, pred_dir=pred_dir_flood, frame_dir=frame_dir_flood,
                                      video_name=name, crop_window=cr_win)
            cfe.run(['extract_trend'], ref_path=ref_path)
