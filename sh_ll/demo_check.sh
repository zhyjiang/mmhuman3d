export CUDA_VISIBLE_DEVICES=3
python demo/estimate_smpl.py \
    configs/hmr/resnet50_hmr_pw3d.py \
    data/checkpoints/resnet50_hmr_pw3d.pth \
    --single_person_demo \
    --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det_checkpoint data/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --input_path  demo/resources/single_person_demo.mp4 \
    --show_path vis_results/single_person_demo.mp4 \
    --output demo_result \
    --smooth_type savgol \
    --speed_up_type deciwatch \
    --draw_bbox


    # python demo/estimate_smpl.py \
    # configs/hmr/resnet50_hmr_pw3d.py \
    # data/checkpoints/resnet50_hmr_pw3d.pth \
    # --multi_person_demo \
    # --tracking_config demo/mmtracking_cfg/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py \
    # --input_path  demo/resources/multi_person_demo.mp4 \
    # --show_path vis_results/multi_person_demo.mp4 \
    # --smooth_type savgol \
    # --speed_up_type deciwatch \
    # # [--draw_bbox]