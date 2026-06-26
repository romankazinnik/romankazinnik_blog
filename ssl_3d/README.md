Instructions:

1. Naive rule-based labeling
python run_pipeline.py --out out_v4_005_strict --e57 ~/Downloads/808_Brannan_LB_E57wP-002-001.e57  --voxel 0.05 --phase 1 --strict

cd out_v4_005_strict

ls

boxes_phase1.json  
boxes_phase1.ply  
classified_phase1.ply  
labels_phase1.json  
remaining_phase1.ply  
working-cloud-instance-phase1.ply  
working-cloud-orig.ply  
working-cloud-semantic-phase1.ply  
working_cloud.ply

cat result.txt  

2. Self-supervised structure model and classifier model-2 

python ../ssl/ssl_scene_probe.py --epochs 40 --device cuda:1 --data working-cloud-instance-phase1.ply --names_json labels_phase1.json  --color_ply working-cloud-orig.ply --out ssl_out_confthresh07_40e_1000K --probe_max_train 1000000 --conf_thresh 0.7 > result.txt

cd ssl_out_confthresh07_40e_1000K  

ls

pred_confidence.ply  
pred_instances.ply  
pred_new_points.ply  
pred_types.ply  
subsampled.ply
