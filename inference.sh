python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_original/ --img_dir ./dataset/S5 
python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_original/ --img_dir ./dataset/S6 
python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_original/ --img_dir ./dataset/S7 
python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_original/ --img_dir ./dataset/S8 
python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_gamma/ --img_dir ./dataset/S5 --use_gamma
python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_gamma/ --img_dir ./dataset/S6 --use_gamma
python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_gamma/ --img_dir ./dataset/S7 --use_gamma
python inference.py --ckpt_path ./checkpoints/best.pth --result_dir ./solution_gamma/ --img_dir ./dataset/S8 --use_gamma