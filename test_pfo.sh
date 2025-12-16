 CUDA_VISIBLE_DEVICES=2 python inference_res_pfo.py \
 --ckpt weight/step=84999_stage2_1_1_4_300.ckpt \
 --config configs/model/stage2/1_1_4/cldm_eps_300_ddim.yaml \
 --json_file_path data/test/kodak/kodak_captions_10.json \
 --output output/pfo/kodak_10 \
 --ddim_steps 3 \
 --ddim_eta 0 \
 --Q 4.0 \
 --add_steps 300