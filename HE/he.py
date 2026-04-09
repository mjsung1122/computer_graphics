import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(org, modified):
    # MSE 계산
    mse = np.mean((org - modified) ** 2)
    # PSNR 계산
    if mse == 0:
        psnr = 100
    else:
        psnr = cv2.PSNR(org, modified)
    # SSIM 계산
    s_score = ssim(org, modified)
    return mse, psnr, s_score

# 1. 이미지 로드 (그레이스케일)
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("이미지 파일을 찾을 수 없습니다.")

# 2. 히스토그램 기법 적용
# (1) Histogram Equalization (HE)
he_img = cv2.equalizeHist(img)

# (2) Adaptive Histogram Equalization (AHE) 
# OpenCV는 기본적으로 CLAHE를 제공하므로, clipLimit을 매우 높게 설정하여 AHE처럼 동작하게 함
ahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
ahe_img = ahe.apply(img)

# (3) Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img)

# 3. 결과 및 히스토그램 시각화
titles = ['Original', 'Equalization (HE)', 'Adaptive (AHE)', 'CLAHE']
images = [img, he_img, ahe_img, clahe_img]

plt.figure(figsize=(16, 10))

for i in range(4):
    # 이미지 출력
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
    
    # 히스토그램 출력
    plt.subplot(2, 4, i + 5)
    plt.hist(images[i].ravel(), 256, [0, 256])
    plt.title(f'{titles[i]} Hist')

plt.tight_layout()
plt.show()

# 4. 성능 지표 계산 및 출력
print(f"{'Technique':<20} | {'MSE':<10} | {'PSNR':<10} | {'SSIM':<10}")
print("-" * 60)

for name, target_img in zip(titles[1:], images[1:]):
    mse_v, psnr_v, ssim_v = calculate_metrics(img, target_img)
    print(f"{name:<20} | {mse_v:>10.2f} | {psnr_v:>10.2f} | {ssim_v:>10.4f}")