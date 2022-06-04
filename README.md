# Sufficient-Vision-Transformer

This is the official github repository for Sufficient Vision Transformer.

For codes for ImageNet experiment, please go to /ImageNet
		
	ImageNet checkpoints available at: https://drive.google.com/drive/folders/1LmLw9Hhx2-uFpAL39k-ptPksjEsM1VWx?usp=sharing
	
	my_main_nrh_t2t_vit4.py: code for training Suf-T2T-ViT-ti.

	my_main_suf_vit_ti.py: code for training Suf-ViT-ti. Corresponding checkpoint name: suf_deit_tiny_case2_arch1.pth

	my_main_suf_vit_ti_arch1.py: code for training Suf-ViT-ti-arch1. Corresponding checkpoint name: Suf_deit_tiny_arch1.pth
	
	my_main_suf_vit_ti_ce.py: code for training Suf-ViT-ti-CE. Corresponding checkpoint: suf_vit_ti_ce.pth

	my_main_suf_vit_ti_srloss_minus.py: code for training Suf-ViT-ti-SR-. Corresponding checkpoint: suf_vit_ti_srloss_minus.pth

	train_suf_pvt.py: code for training Suf-PVT-ti. Corresponding checkpoint: suf_pvt_tiny.pth

	train_suf_pvt_small.py: code for training Suf-PVT-small. Corresponding checkpoint: suf_pvt_small.pth

	train_suf_swin_xtiny_case2.py: code for training Suf-Swin-xti. Corresponding checkpoint: suf_swin_xtiny_case2.pth

	train_swin_xtiny.py: code for training Swin-xti. Corresponding checkpoint: swin_xtiny.pth

	test_cifar_ae.py: code for testing Suf-ViT-ti (or other models) under adversarial attack or inputs corrupted by CIFAR-10 images.

For experiments on CIFAR-10 ablation study, please go to /CIFAR_10

	aug_vit_base.py: Training deit-base on CIFAR-10
	
	aug_vit_small.py: Training deit-small on CIFAR-10
	
	sufvit_s.py: Training Suf-ViT-small on CIFAR-10
	
	sufvit_b.py: Training Suf-ViT-base on CIFAR-10
	
	info_sufvit_s.py: Calculate mutual information between Suf-ViT-small input and hidden layer outputs at different depths. Need to run after running "sufvit_s.py".
	
	info_sufvit_s_y.py: Calculate mutual information between Suf-ViT-small input label and hidden layer outputs at different depths. Need to run after running "sufvit_s.py".
	
	info_vit_small.py: Calculate mutual information between DeiT-small input and hidden layer outputs at different depths. Need to run after running "aug_vit_small.py".
	
	info_vit_small_y.py: Calculate mutual information between DeiT-small input label and hidden layer outputs at different depths. Need to run after running "aug_vit_small.py".
	
	Suf-ViT-x.py and DeiT-tiny.py: Codes for ablation study on S-Block architecture
