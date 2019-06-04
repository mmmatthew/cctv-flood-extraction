# cctv-flood-extraction
Framework to extract a flooding trend out of cctv videos by using pre-trained CNNs.

In the `scripts` directory you will find:
- `train_unet.py`: script for training the DCNNs
- `fine-tune_unet.py`: script for fine-tuning the DCNN to individual videos
- `test-unet.py`: script for evaluating the segmentation performance of DCNNs
- `soficorrelation.py`: script that loops through video frames, performing segmentation and extracting the SOFI index

In the `sofi_extraction` directory you will find `engine.py`, which contains the class used to set up and manage the neural network.