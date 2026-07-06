# ImputeECG: Deep Learning Reconstruction of Complete 12-Lead Electrocardiograms from Incomplete Recordings for Cardiac Assessment

### Abstract: 
Complete digital 12-lead electrocardiograms (ECGs) are essential for AI-enabled cardiovascular assessment, yet many clinical ECG records, particularly those digitized from ECG images, remain incomplete because of short display formats, incomplete waveform digitization, lead loss, or signal corruption. We developed ImputeECG, a mask-conditioned one-dimensional Transformer autoencoder that completes 12-lead, 10-s ECGs while retaining all observed samples. The model was trained on PTB-XL and evaluated on PTB-XL and CPSC2018 under simulated incomplete settings, with additional real-world validation in a 43{,}633-record Kailuan clinical cohort after ECG image digitization. Metrics were computed over originally missing regions, with analyses of morphology and downstream diagnostic utility. On PTB-XL, ImputeECG reduced missing-region MAE by 41.7--51.0\% and MSE by 54.0--63.7\% versus the strongest baseline, with lower errors in R-peak timing, RR interval, QRS duration, QT interval, and P-wave, QRS-complex, and T-wave reconstruction. On CPSC2018, ImputeECG reduced MAE by 49.7--51.9\%, supporting external generalization. In downstream multi-label classification, ImputeECG restored performance to 92.28\% AUROC and 33.88\% AUPRC in the most incomplete PTB-XL setting, approaching complete-ECG performance. On CPSC2018, completed ECGs achieved 94.75--95.89\% AUROC and 78.83--81.86\% AUPRC across settings. In Kailuan, ECG completion improved zero-shot sex prediction AUROC from 82.6\% to 85.8\% and reduced age prediction MAE from 10.72 to 9.87 years after image-based ECG digitization. These findings support ECG completion as a practical strategy for converting incomplete ECG records into AI-ready 12-lead, 10-s digital signals and extending the usable scope of ECG archives for digital cardiac assessment.

## ImputeECG Training
```sh
python train.py \
  --data_path ../data/ptbxl \
  --output_dir ./output_dir \
  --log_dir ./output_dir \
  --gpu 0 \
  --epochs 100 \
  --batch_size 128 \
  --mask_ratio 0.20 \
  --missing_sentinel 65535
```

## ImputeECG Inference
```sh
python inference.py \
  --checkpoint ./checkpoints/checkpoint-100.pth \
  --device cuda:0 \
  --missing_sentinel 65535
```
## ImputeECG Checkpoint
```sh
https://drive.google.com/drive/folders/1RiwQPN4_of6p5wsj8qUa1D6IYW_VKAxX
```
