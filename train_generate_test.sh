experiment="MFA-0.0001tv_0.25nms_abs_obj+cls"

python train_mfa.py --device 0 --experiment $experiment
python generate.py --experiment $experiment

source /home/chenwen/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab
img_root="/home/chenwen/Code/MFA/savedImage/"
cd ../mmdetection
python test_digital.py configs/ssd/ssd512_coco.py  weights/ssd512_coco_20210803_022849-0a47a1ca.pth --img_path $img_root$experiment
python test_digital.py configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py weights/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth --img_path $img_root$experiment
python test_digital.py configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py weights/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth --img_path $img_root$experiment
python test_digital.py configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py weights/cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth --img_path $img_root$experiment
python test_digital.py configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py weights/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth --img_path $img_root$experiment
python test_digital.py configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py weights/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth --img_path $img_root$experiment
python test_digital.py configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py weights/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20211210_213728-4a824142.pth --img_path $img_root$experiment
python test_digital.py configs/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py weights/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth --img_path $img_root$experiment
cd ../yolov5
python test_digital.py --source $img_root$experiment
cd ../yolov7
python test_digital.py --source $img_root$experiment
