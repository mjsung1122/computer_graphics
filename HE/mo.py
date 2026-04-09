import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 로드 및 이진화 (그레이스케일 -> 임계값 처리)
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    # 이미지 파일이 없을 경우 테스트용 더미 이미지 생성
    img = np.zeros((300, 300), dtype="uint8")
    cv2.putText(img, "ABC", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 5)

_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 2. 커널(Structuring Element) 설정 (보통 3x3 또는 5x5 사각형)
kernel = np.ones((5, 5), np.uint8)

# 3. 형태학적 연산 적용
dilation = cv2.dilate(binary_img, kernel, iterations=1)  # 팽창
erosion = cv2.erode(binary_img, kernel, iterations=1)    # 침식
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)   # 열기 (침식 후 팽창: 노이즈 제거)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)  # 닫기 (팽창 후 침식: 구멍 채우기)

# 4. 시각화
titles = ['Binary Original', 'Dilation', 'Erosion', 'Opening', 'Closing']
images = [binary_img, dilation, erosion, opening, closing]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()