# Encoder and resource index

This index records the encoder currently required by each published experiment. Weight paths are copied from the release YAMLs. All model weights and pretrained snapshots are external inputs.

The exact per-configuration mappings are in [experiment-index.csv](experiment-index.csv) and [experiment-index.json](experiment-index.json).

## Encoders

| Encoder ID | Model | Training configuration | Expected evaluation weight path | Evaluation YAMLs |
|---|---|---|---|---|
| `vjepa_agibot_time` | V-JEPA agibot Time-JEPA | [configs/train_time_jepa/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_time_jepa_conv1x1_final_wholebody_cartesian_v2_15000.yaml](../configs/train_time_jepa/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_time_jepa_conv1x1_final_wholebody_cartesian_v2_15000.yaml) | `outputs/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_time_jepa_conv1x1_final_wholebody_cartesian_v2_lr3e-5_e20_n15000/last.pth.tar` | 7 |
| `vjepa_agibot_video_only` | V-JEPA agibot Video FT | [configs/train_time_jepa/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_video_only_final_15000.yaml](../configs/train_time_jepa/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_video_only_final_15000.yaml) | `outputs/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_video_only_lr3e-5_e20_n15000/last.pth.tar` | 3 |
| `vjepa_agibot_covla_egoexo_time` | V-JEPA agibot, covla, egoexo Time-JEPA | [configs/train_time_jepa/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_time_jepa_seed239.yaml](../configs/train_time_jepa/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_time_jepa_seed239.yaml) | `outputs/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_time_jepa_seed239/last.pth.tar` | 24 |
| `vjepa_agibot_covla_egoexo_video_only` | V-JEPA agibot, covla, egoexo Video FT | [configs/train_time_jepa/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_video_only_seed239.yaml](../configs/train_time_jepa/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_video_only_seed239.yaml) | `outputs/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_video_only_seed239/last.pth.tar` | 12 |
| `vjepa_egoexo_matched_time` | V-JEPA human motion matched single-domain Time-JEPA | [configs/train_time_jepa/covla_egoexo/egoexo_only_exo124_view3ood_vitg384_64f5fps_time_jepa_seed239.yaml](../configs/train_time_jepa/covla_egoexo/egoexo_only_exo124_view3ood_vitg384_64f5fps_time_jepa_seed239.yaml) | `outputs/covla_egoexo/egoexo_only_exo124_view3ood_vitg384_64f5fps_time_jepa_seed239/last.pth.tar` | 2 |
| `vjepa_covla_egoexo_time` | V-JEPA covla, egoexo Time-JEPA | [configs/train_time_jepa/covla_egoexo/mix_exposure_covla_egoexo_vitg384_64f5fps_time_jepa_seed239.yaml](../configs/train_time_jepa/covla_egoexo/mix_exposure_covla_egoexo_vitg384_64f5fps_time_jepa_seed239.yaml) | `outputs/covla_egoexo/mix_exposure_covla_egoexo_vitg384_64f5fps_time_jepa_seed239/last.pth.tar` | 2 |
| `vjepa_egoexo_time` | V-JEPA egoexo Time-JEPA | [configs/train_time_jepa/egoexo/egoexo_exo124_view3ood_vitg384_64f_time_jepa_conv1x1_final_whole_body_compact_v2.yaml](../configs/train_time_jepa/egoexo/egoexo_exo124_view3ood_vitg384_64f_time_jepa_conv1x1_final_whole_body_compact_v2.yaml) | `outputs/egoexo/egoexo_exo124_view3ood_vitg384_64f_time_jepa_conv1x1_final_whole_body_compact_v2_lr3e-5_e20/last.pth.tar` | 8 |
| `vjepa_egoexo_video_only` | V-JEPA egoexo Video FT | [configs/train_time_jepa/egoexo/egoexo_exo124_view3ood_vitg384_64f_video_only.yaml](../configs/train_time_jepa/egoexo/egoexo_exo124_view3ood_vitg384_64f_video_only.yaml) | `outputs/egoexo/egoexo_exo124_view3ood_vitg384_64f_video_only_lr3e-5_e20/last.pth.tar` | 6 |
| `vjepa_egoexo_a4_m2c_1d_pelvis_speed` | V-JEPA human motion a4_m2c_1d_pelvis_speed | [configs/train_time_jepa/egoexo_ablation/egoexo_a4_m2c_1d_pelvis_speed.yaml](../configs/train_time_jepa/egoexo_ablation/egoexo_a4_m2c_1d_pelvis_speed.yaml) | `outputs/egoexo_ablation/exo124_view3ood/a4_m2c_1d_pelvis_speed/encoder_seed239_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_a4_m2c_5d_motion_only` | V-JEPA human motion a4_m2c_5d_motion_only | [configs/train_time_jepa/egoexo_ablation/egoexo_a4_m2c_5d_motion_only.yaml](../configs/train_time_jepa/egoexo_ablation/egoexo_a4_m2c_5d_motion_only.yaml) | `outputs/egoexo_ablation/exo124_view3ood/a4_m2c_5d_motion_only/encoder_seed239_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_a4_m2d_take_derangement` | V-JEPA human motion a4_m2d_take_derangement | [configs/train_time_jepa/egoexo_ablation/egoexo_a4_m2d_take_derangement.yaml](../configs/train_time_jepa/egoexo_ablation/egoexo_a4_m2d_take_derangement.yaml) | `outputs/egoexo_ablation/exo124_view3ood/a4_m2d_take_derangement/encoder_seed239_lr3e-5_e20/last.pth.tar` | 1 |
| `vjepa_egoexo_head_h0_conv1x1` | V-JEPA human motion head h0_conv1x1 | [configs/train_time_jepa/egoexo_head_sweep/h0_conv1x1.yaml](../configs/train_time_jepa/egoexo_head_sweep/h0_conv1x1.yaml) | `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h0_conv1x1_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_head_h2_large_k3` | V-JEPA human motion head h2_large_k3 | [configs/train_time_jepa/egoexo_head_sweep/h2_large_k3.yaml](../configs/train_time_jepa/egoexo_head_sweep/h2_large_k3.yaml) | `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h2_large_k3_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_head_h3_strong_dense_fusion` | V-JEPA human motion head h3_strong_dense_fusion | [configs/train_time_jepa/egoexo_head_sweep/h3_strong_dense_fusion.yaml](../configs/train_time_jepa/egoexo_head_sweep/h3_strong_dense_fusion.yaml) | `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h3_strong_dense_fusion_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_head_h4_global_attention` | V-JEPA human motion head h4_global_attention | [configs/train_time_jepa/egoexo_head_sweep/h4_global_attention.yaml](../configs/train_time_jepa/egoexo_head_sweep/h4_global_attention.yaml) | `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h4_global_attention_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_head_h4_local_attention` | V-JEPA human motion head h4_local_attention | [configs/train_time_jepa/egoexo_head_sweep/h4_local_attention.yaml](../configs/train_time_jepa/egoexo_head_sweep/h4_local_attention.yaml) | `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h4_local_attention_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_head_h4_temporal_attention` | V-JEPA human motion head h4_temporal_attention | [configs/train_time_jepa/egoexo_head_sweep/h4_temporal_attention.yaml](../configs/train_time_jepa/egoexo_head_sweep/h4_temporal_attention.yaml) | `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h4_temporal_attention_lr3e-5_e20/last.pth.tar` | 2 |
| `vjepa_egoexo_gaussian_noise_1p0sigma` | V-JEPA human motion gaussian_noise_1p0sigma | [configs/train_time_jepa/egoexo_sensor_robustness/egoexo_gaussian_noise_1p0sigma.yaml](../configs/train_time_jepa/egoexo_sensor_robustness/egoexo_gaussian_noise_1p0sigma.yaml) | `outputs/egoexo_sensor_robustness/exo124_view3ood/gaussian_noise_1p0sigma/encoder_seed239_lr3e-5_e20/last.pth.tar` | 1 |
| `vjepa_egoexo_time_offset_0p8s` | V-JEPA human motion time_offset_0p8s | [configs/train_time_jepa/egoexo_sensor_robustness/egoexo_time_offset_0p8s.yaml](../configs/train_time_jepa/egoexo_sensor_robustness/egoexo_time_offset_0p8s.yaml) | `outputs/egoexo_sensor_robustness/exo124_view3ood/time_offset_0p8s/encoder_seed239_lr3e-5_e20/last.pth.tar` | 1 |
| `vjepa_covla_time` | V-JEPA covla Time-JEPA | [configs/train_time_jepa/weather_ood/covla_vitg384_64f_time_jepa_conv1x1_final.yaml](../configs/train_time_jepa/weather_ood/covla_vitg384_64f_time_jepa_conv1x1_final.yaml) | `outputs/weather_ood/covla_vitg384_64f5fps_strict_time_jepa_conv1x1_final_split6305_lr3e-5_e20/last.pth.tar` | 7 |
| `vjepa_covla_time_law` | V-JEPA autonomous driving Time-JEPA + Law | [configs/train_time_jepa/weather_ood/covla_vitg384_64f_time_jepa_law_alpha0p5.yaml](../configs/train_time_jepa/weather_ood/covla_vitg384_64f_time_jepa_law_alpha0p5.yaml) | `outputs/weather_ood/covla_time_jepa_law_video_pts_v2_alpha0p5_seed239_e20/last.pth.tar` | 5 |
| `vjepa_covla_video_only` | V-JEPA covla Video FT | [configs/train_time_jepa/weather_ood/covla_vitg384_64f_video_only.yaml](../configs/train_time_jepa/weather_ood/covla_vitg384_64f_video_only.yaml) | `outputs/weather_ood/covla_vitg384_64f5fps_strict_video_only_split6305_lr3e-5_e20/last.pth.tar` | 3 |
| `psg_agibot_full` | PSG-JEPA ViT-g full | [external_baselines/psg_jepa/configs/agibot_full_vitg_n15000.yaml](../external_baselines/psg_jepa/configs/agibot_full_vitg_n15000.yaml) | `outputs_external/psg_jepa_vitg/agibot_full_g_port_seed239_n15000/last.pt` | 4 |
| `psg_agibot_static` | PSG-JEPA ViT-g static | [external_baselines/psg_jepa/configs/agibot_static_vitg_n15000.yaml](../external_baselines/psg_jepa/configs/agibot_static_vitg_n15000.yaml) | `outputs_external/psg_jepa_vitg/agibot_static_g_port_seed239_n15000/last.pt` | 4 |
| `ivideogpt_agibot_time` | iVideoGPT agibot Time-JEPA | [world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/agibot/time.yaml](../world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/agibot/time.yaml) | `outputs_world_model/ivideogpt_oxe_64_medium/time/agibot_n15000/encoder/checkpoints/last/transformer/model.safetensors` | 2 |
| `ivideogpt_agibot_video_only` | iVideoGPT agibot Video FT | [world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/agibot/video_only.yaml](../world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/agibot/video_only.yaml) | `outputs_world_model/ivideogpt_oxe_64_medium/video_only/agibot_n15000/encoder/checkpoints/last/transformer/model.safetensors` | 2 |
| `ivideogpt_covla_time` | iVideoGPT covla Time-JEPA | [world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/covla/time.yaml](../world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/covla/time.yaml) | `outputs_world_model/ivideogpt_oxe_64_medium/time/covla/encoder/checkpoints/last/transformer/model.safetensors` | 2 |
| `ivideogpt_covla_video_only` | iVideoGPT covla Video FT | [world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/covla/video_only.yaml](../world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/covla/video_only.yaml) | `outputs_world_model/ivideogpt_oxe_64_medium/video_only/covla/encoder/checkpoints/last/transformer/model.safetensors` | 2 |
| `ivideogpt_egoexo_time` | iVideoGPT egoexo Time-JEPA | [world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/egoexo/time.yaml](../world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/egoexo/time.yaml) | `outputs_world_model/ivideogpt_oxe_64_medium/time/egoexo/encoder/checkpoints/last/transformer/model.safetensors` | 2 |
| `ivideogpt_egoexo_video_only` | iVideoGPT egoexo Video FT | [world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/egoexo/video_only.yaml](../world_model_probe_rjob/configs/ivideogpt_oxe_64_medium/adaptation/egoexo/video_only.yaml) | `outputs_world_model/ivideogpt_oxe_64_medium/video_only/egoexo/encoder/checkpoints/last/transformer/model.safetensors` | 2 |
| `vjepa_original` | Original V-JEPA 2.1 ViT-g/16 | Official pretrained model | `vjepa2_1_vitg_384.pt` | 18 |
| `internvideo_original` | InternVideo-Next-L frozen | Official pretrained model | `InternVideo/checkpoints/internvideo_next_large_p14_res224_f16/model.safetensors` | 6 |
| `videomaev2_original` | VideoMAE V2-G frozen | Official pretrained model | `VideoMAEv2/checkpoints/mae-g/vit_g_hybrid_pt_1200e.pth` | 6 |
| `ivideogpt_original` | iVideoGPT OXE frozen | Official pretrained model | `iVideoGPT/checkpoints/ivideogpt-oxe-64-act-free-medium/transformer/model.safetensors` | 6 |

## Resources

`included` means the path belongs to the public release manifest, regardless of local links used for acceptance. `external` means data, weights, URDF or upstream sources must be obtained separately. `generate` means run the listed preparation entrypoint before evaluation.

| Resource | Kind | Availability | Preparation / note |
|---|---|---|---|
| `data/CoVLA-Dataset` | dataset | external | [time-jepa/make_covla_weather_ood_splits.py](../time-jepa/make_covla_weather_ood_splits.py) |
| `data/EgoExo4D_minimal` | dataset | external | [time-jepa/make_egoexo_splits.py](../time-jepa/make_egoexo_splits.py) |
| `data/agibot` | dataset | external | [time-jepa/make_agibot_splits.py](../time-jepa/make_agibot_splits.py) |
| `data/physion_v1_5` | dataset | external | [physion_ocp_rjob/check_environment.py](../physion_ocp_rjob/check_environment.py) |
| `external_baselines/grasp/data/videos/level2` | dataset | external | [grasp_eval_rjob/audit_data.py](../grasp_eval_rjob/audit_data.py) |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/agibot_n15000/encoder/checkpoints/last/adaptation_manifest.json` | encoder_metadata | external | Adapter-required model metadata |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/agibot_n15000/encoder/checkpoints/last/transformer/config.json` | encoder_metadata | external | Adapted transformer configuration |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/covla/encoder/checkpoints/last/adaptation_manifest.json` | encoder_metadata | external | Adapter-required model metadata |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/covla/encoder/checkpoints/last/transformer/config.json` | encoder_metadata | external | Adapted transformer configuration |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/egoexo/encoder/checkpoints/last/adaptation_manifest.json` | encoder_metadata | external | Adapter-required model metadata |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/egoexo/encoder/checkpoints/last/transformer/config.json` | encoder_metadata | external | Adapted transformer configuration |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/agibot_n15000/encoder/checkpoints/last/adaptation_manifest.json` | encoder_metadata | external | Adapter-required model metadata |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/agibot_n15000/encoder/checkpoints/last/transformer/config.json` | encoder_metadata | external | Adapted transformer configuration |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/covla/encoder/checkpoints/last/adaptation_manifest.json` | encoder_metadata | external | Adapter-required model metadata |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/covla/encoder/checkpoints/last/transformer/config.json` | encoder_metadata | external | Adapted transformer configuration |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/egoexo/encoder/checkpoints/last/adaptation_manifest.json` | encoder_metadata | external | Adapter-required model metadata |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/egoexo/encoder/checkpoints/last/transformer/config.json` | encoder_metadata | external | Adapted transformer configuration |
| `InternVideo/checkpoints/internvideo_next_large_p14_res224_f16/model.safetensors` | encoder_weight | external |  |
| `VideoMAEv2/checkpoints/mae-g/vit_g_hybrid_pt_1200e.pth` | encoder_weight | external |  |
| `iVideoGPT/checkpoints/ivideogpt-oxe-64-act-free-medium/transformer/model.safetensors` | encoder_weight | external |  |
| `outputs/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_time_jepa_conv1x1_final_wholebody_cartesian_v2_lr3e-5_e20_n15000/last.pth.tar` | encoder_weight | external |  |
| `outputs/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_video_only_lr3e-5_e20_n15000/last.pth.tar` | encoder_weight | external |  |
| `outputs/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_time_jepa_seed239/last.pth.tar` | encoder_weight | external |  |
| `outputs/covla_agibot_egoexo/mix_exposure_covla_agibot15000_egoexo_vitg384_64f5fps_video_only_seed239/last.pth.tar` | encoder_weight | external |  |
| `outputs/covla_egoexo/egoexo_only_exo124_view3ood_vitg384_64f5fps_time_jepa_seed239/last.pth.tar` | encoder_weight | external |  |
| `outputs/covla_egoexo/mix_exposure_covla_egoexo_vitg384_64f5fps_time_jepa_seed239/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo/egoexo_exo124_view3ood_vitg384_64f_time_jepa_conv1x1_final_whole_body_compact_v2_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo/egoexo_exo124_view3ood_vitg384_64f_video_only_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_ablation/exo124_view3ood/a4_m2c_1d_pelvis_speed/encoder_seed239_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_ablation/exo124_view3ood/a4_m2c_5d_motion_only/encoder_seed239_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_ablation/exo124_view3ood/a4_m2d_take_derangement/encoder_seed239_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h0_conv1x1_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h2_large_k3_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h3_strong_dense_fusion_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h4_global_attention_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h4_local_attention_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_head_sweep/exo124_view3ood_whole9/h4_temporal_attention_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_sensor_robustness/exo124_view3ood/gaussian_noise_1p0sigma/encoder_seed239_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/egoexo_sensor_robustness/exo124_view3ood/time_offset_0p8s/encoder_seed239_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/weather_ood/covla_time_jepa_law_video_pts_v2_alpha0p5_seed239_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/weather_ood/covla_vitg384_64f5fps_strict_time_jepa_conv1x1_final_split6305_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs/weather_ood/covla_vitg384_64f5fps_strict_video_only_split6305_lr3e-5_e20/last.pth.tar` | encoder_weight | external |  |
| `outputs_external/psg_jepa_vitg/agibot_full_g_port_seed239_n15000/last.pt` | encoder_weight | external |  |
| `outputs_external/psg_jepa_vitg/agibot_static_g_port_seed239_n15000/last.pt` | encoder_weight | external |  |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/agibot_n15000/encoder/checkpoints/last/transformer/model.safetensors` | encoder_weight | external |  |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/covla/encoder/checkpoints/last/transformer/model.safetensors` | encoder_weight | external |  |
| `outputs_world_model/ivideogpt_oxe_64_medium/time/egoexo/encoder/checkpoints/last/transformer/model.safetensors` | encoder_weight | external |  |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/agibot_n15000/encoder/checkpoints/last/transformer/model.safetensors` | encoder_weight | external |  |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/covla/encoder/checkpoints/last/transformer/model.safetensors` | encoder_weight | external |  |
| `outputs_world_model/ivideogpt_oxe_64_medium/video_only/egoexo/encoder/checkpoints/last/transformer/model.safetensors` | encoder_weight | external |  |
| `vjepa2_1_vitg_384.pt` | encoder_weight | external |  |
| `requirements-ivideogpt.txt` | environment | included | Additional iVideoGPT dependencies |
| `requirements-psg.txt` | environment | included | Additional PSG training dependencies |
| `requirements.txt` | environment | included | Install the recorded jepa package versions |
| `grasp_eval_rjob/artifacts/data_audit.json` | gate_or_audit | generate | [grasp_eval_rjob/audit_data.py](../grasp_eval_rjob/audit_data.py) |
| `physics_consistency/agibot_reach_rate/results/gt_gate/gt_gate_metrics.json` | gate_or_audit | generate | [physics_consistency/agibot_reach_rate/gt_gate.py](../physics_consistency/agibot_reach_rate/gt_gate.py) |
| `physics_consistency/covla/results/gt_gate/gt_gate_metrics.json` | gate_or_audit | generate | [physics_consistency/covla/covla_gt_consistency_gate.py](../physics_consistency/covla/covla_gt_consistency_gate.py) |
| `physics_consistency/egoexo/results/gt_gate/gt_gate_metrics.json` | gate_or_audit | generate | [physics_consistency/egoexo/gt_kinematic_gate.py](../physics_consistency/egoexo/gt_kinematic_gate.py) |
| `physics_consistency/egoexo_knee/results/gt_gate/gt_gate_metrics.json` | gate_or_audit | generate | [physics_consistency/egoexo_knee/gt_kinematic_gate.py](../physics_consistency/egoexo_knee/gt_kinematic_gate.py) |
| `time-jepa/cache/covla_law_train6305_gate_video_pts_v2.json` | gate_or_audit | generate | [covla_law_rjob/gate.py](../covla_law_rjob/gate.py) |
| `configs/train_time_jepa/agibot/agibot_3view_top_head_hand_left_hand_right_vitg384_64f_time_jepa_conv1x1_final_wholebody_cartesian_v2_15000.yaml` | input_config | included |  |
| `physics_consistency/agibot_reach_rate/results/cache/agibot_q14_endpoint_v6_probe_train_stats_seed239.json` | normalization | included |  |
| `physics_consistency/covla/results/cache/covla_4fps_train_stats_seed239.json` | normalization | generate | [physics_consistency/covla/train_independent_head_bank.py](../physics_consistency/covla/train_independent_head_bank.py) |
| `physics_consistency/egoexo/results/cache/egoexo_arm4_probe_train_stats_seed239.json` | normalization | generate | [physics_consistency/egoexo/train_independent_head_bank.py](../physics_consistency/egoexo/train_independent_head_bank.py) |
| `physics_consistency/egoexo_knee/results/cache/egoexo_knee4_probe_train_stats_seed239.json` | normalization | generate | [physics_consistency/egoexo_knee/train_independent_head_bank.py](../physics_consistency/egoexo_knee/train_independent_head_bank.py) |
| `time-jepa/cache/agibot_3view_top_head_hand_left_hand_right_wholebody_cartesian_v2_64f5fps_stats.json` | normalization | included |  |
| `time-jepa/cache/agibot_psg_action40_5fps_grid_stats.json` | normalization | included |  |
| `time-jepa/cache/covla_weather_ood_encoder_train_6305_64f5fps_strict_vEgo_aEgo_yaw_rate_stats.json` | normalization | generate | [covla_jepa_rjob/train_from_yaml.py](../covla_jepa_rjob/train_from_yaml.py) |
| `time-jepa/cache/egoexo_exo124_view3ood_m2c_1d_pelvis_speed_64f5fps_stats.json` | normalization | generate | [covla_jepa_rjob/train_from_yaml.py](../covla_jepa_rjob/train_from_yaml.py) |
| `time-jepa/cache/egoexo_exo124_view3ood_m2c_5d_motion_only_64f5fps_stats.json` | normalization | generate | [covla_jepa_rjob/train_from_yaml.py](../covla_jepa_rjob/train_from_yaml.py) |
| `time-jepa/cache/egoexo_exo124_view3ood_whole_body_compact_v2_64f5fps_stats.json` | normalization | generate | [covla_jepa_rjob/train_from_yaml.py](../covla_jepa_rjob/train_from_yaml.py) |
| `time-jepa/cache/agibot_wholebody_cartesian_v1_waist_pca.json` | pca | included | [time-jepa/fit_agibot_waist_pca.py](../time-jepa/fit_agibot_waist_pca.py) |
| `iVideoGPT/checkpoints/ivideogpt-oxe-64-act-free-medium` | pretrained_snapshot | external |  |
| `InternVideo/checkpoints/internvideo_next_large_p14_res224_f16` | snapshot_companions | external | Full pretrained snapshot with custom modeling and configuration Python files |
| `iVideoGPT/checkpoints/ivideogpt-oxe-64-act-free-medium` | snapshot_companions | external | Full official tokenizer/transformer snapshot and configuration files |
| `VideoMAEv2` | source_checkout | external | [scripts/setup_third_party.py](../scripts/setup_third_party.py) |
| `iVideoGPT` | source_checkout | external | [scripts/setup_third_party.py](../scripts/setup_third_party.py) |
| `third_party/PSG-JEPA` | source_checkout | external | [scripts/setup_third_party.py](../scripts/setup_third_party.py) |
| `time-jepa/splits` | split_metadata | included | [time-jepa/make_agibot_splits.py](../time-jepa/make_agibot_splits.py) |
| `time-jepa/splits/agibot_top_head_encoder_train.json` | split_metadata | included |  |
| `time-jepa/splits/covla_weather_ood` | split_metadata | generate | [scripts/prepare_metadata.py](../scripts/prepare_metadata.py) |
| `time-jepa/splits/covla_weather_ood/covla_encoder_train.json` | split_metadata | generate | [scripts/prepare_metadata.py](../scripts/prepare_metadata.py) |
| `time-jepa/splits/egoexo_exo124_view3ood_encoder_train.json` | split_metadata | generate | [scripts/prepare_metadata.py](../scripts/prepare_metadata.py) |
| `time-jepa/splits/egoexo_exo124_view3ood_split_manifest.json` | split_metadata | generate | [scripts/prepare_metadata.py](../scripts/prepare_metadata.py) |
| `physics_consistency/agibot/assets/G2_NO_GRIPPER.urdf` | urdf | external |  |

## Path consistency notes

All indexed training output directories match their corresponding evaluation encoder input directories.

The current YAMLs are unchanged by indexing. Adapted iVideoGPT rows identify the adapted encoder as primary; the official snapshot is a supporting input, not a second compared model. For iVideoGPT, retain the snapshot configuration/tokenizer files and adaptation manifest required by its adapter. GRASP also needs the predictor and target encoder contained in its configured checkpoint. Dataset preparation and omitted URDF/GT-gate resources are described in [data.md](data.md); third-party installation is in [dependencies.md](dependencies.md).
