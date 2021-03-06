# Sufficient-Vision-Transformer

This is the official github repository for Sufficient Vision Transformer.

# Introduction
Currently, Vision Transformer (ViT) and its variants have demonstrated promising performance on various computer vision tasks. Nevertheless, task-irrelevant information such as background nuisance and noise in patch tokens would damage the performance of ViT-based models. In this paper, we develop Sufficient Vision Transformer (Suf-ViT) as a new solution to address this issue. In our research, we propose the Sufficiency-Blocks (S-Blocks) to be applied across the depth of Suf-ViT to disentangle and discard task-irrelevant information accurately. Besides, to boost the training of Suf-ViT, we formulate a Sufficient-Reduction Loss (SRLoss) leveraging the concept of Mutual Information (MI) that enables Suf-ViT to extract more reliable sufficient representations by removing task-irrelevant information. 

# Method
To boost ViT-based model robustness and performance, we propose the design of Sufficient Vision Transformer (Suf-ViT). In detail, our Suf-ViT is implemented with Sufficiency-Blocks (S-Block) to maximally remove task-irrelevant information by encoding it into token sequence x_G^i (where G stands for task-irrelevant information, 𝑖 is the stage that the S-block is in) while passes task-relevant information forward into the next stage by encoding it into token sequence x_O^i (where O stands for output, 𝑖 is the stage that the S-block is in). Besides, we proposed Sufficient Reduction Loss (SRLoss) as regularization to facilitate the training of Suf-ViT and remove task-irrelevant information in general. With these settings, Suf-ViT encodes a sufficient representation of the input, which contains complete details
on the task but has task-irrelevant information maximally removed. By making predictions based on such sufficient representations, the performance of Suf-ViT can be boosted by having less influence by task-irrelevant information.

The figure below demonstrates the overall architecture of a Sufficient Vision Transformer
![alt text](https://github.com/zhicheng2T0/Sufficient-Vision-Transformer/blob/main/sufvit2.PNG)

The figure below demonstrates the architecture of a Sufficiency Block
![alt text](https://github.com/zhicheng2T0/Sufficient-Vision-Transformer/blob/main/sblock.PNG)

The formulation of Sufficient Reduction Loss is demonstrated below
![alt text](https://github.com/zhicheng2T0/Sufficient-Vision-Transformer/blob/main/srloss.PNG)

# Key results

Key results on ImageNet:

	Suf-T2T-ViT-ti (FLOPs=1.2B, Params=7.6M), Top1-acc=73.3%

	Suf-ViT-ti (FLOPs=1.24B, Params=11.5M), Top1-acc=75.6%
	
	Suf-PVT-ti (FLOPs=1.9B, Params=13.2M), Top1-acc=75.6%

Performance of corresponding baselines:

	T2T-ViT-7 (FLOPs=1.2B, Params=4.3M), Top1-acc=71.7%	
	
	DeiT-ti (FLOPs=1.3B, Params=5.7M), Top1-acc=72.2%
	
	PVT-ti (FLOPs=1.9B, Params=13.2M), Top1-acc=74.8%
	
Key results on ImageNet-C (mCE indicates Mean Corruption Error, lower mCE indicates better robustness):

	DeiT-ti (FLOPs=1.2B, Params=4.3M), Top1-acc=72.2%, mCE=72.9%
	
	KVT-ti  (FLOPs=1.3B, Params=5.7M), Top1-acc=73.0%, mCE=72.0%
	
	DWT-ViT-ti  (FLOPs=1.4B, Params=10.6M), Top1-acc=74.2%, mCE=69.0%
	
	PVT-ti  (FLOPs=1.9B, Params=13.2M), Top1-acc=74.8%, mCE=77.7%
	
	ConViT-ti  (FLOPs=1.4B, Params=5.7M), Top1-acc=73.3%, mCE=68.4%
	
	DeiT-ti-Augmix  (FLOPs=1.3B, Params=5.7M), Top1-acc=72.4%, mCE=67.7%
	
	Suf-ViT-ti  (FLOPs=1.24B, Params=11.5M), Top1-acc=75.6%, mCE=63.6%

Performance of Suf-ViT and various baseline models on ImageNet corrupted by CIFAR-10 images as corruption at different norms:

![alt text](https://github.com/zhicheng2T0/Sufficient-Vision-Transformer/blob/main/cifar_corruption.PNG)

Performance of Suf-ViT and various baseline models on ImageNet corrupted by adversarial noise (generated using FGSM) at different norms:

![alt text](https://github.com/zhicheng2T0/Sufficient-Vision-Transformer/blob/main/fgsm_corruption.PNG)


# Acknowledgement

This work was supported in part by the Australian Research Council under Project DP210101859 and the University of Sydney SOAR Prize.


# Citation
If you find our work to be useful for your research, please consider citing:

	@InProceedings{sufvit,
		author    = {Cheng, Zhi and Su, Xiu and Wang, Xueyu and You, Shan and Xu, Chang},
		title     = {Sufficient Vision Transformer},
		booktitle = {28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), Washington, DC, USA, ACM, New York, NY, USA, 11 pages},
		url       = {https://doi.org/10.1145/3534678.3539322},
		month     = {August 14-18},
		year      = {2022}
	}


# Train Models from scratch

For codes on ImageNet experiments, please go to /ImageNet
		
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
