import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 로드 (연산을 위해 같은 크기의 이미지 2개가 필요함)
# 여기서는 원본 이미지와 동일한 크기의 '연산용 이미지'를 생성해서 사용하겠습니다.
img1 = cv2.imread('input.jpg')
if img1 is None:
    raise FileNotFoundError("input.jpg 파일을 찾을 수 없습니다.")

# 연산을 위해 img1과 동일한 크기의 밝은 회색 이미지(값 50) 생성
img2 = np.full_like(img1, 50) 

# 2. 산술 연산 수행
# (1) 덧셈: 이미지가 전반적으로 밝아짐
added = cv2.add(img1, img2)

# (2) 뺄셈: 이미지가 전반적으로 어두워짐
subtracted = cv2.subtract(img1, img2)

# (3) 곱셈: 대비가 강해지며 밝은 영역은 쉽게 화이트아웃됨
# 곱셈은 스케일링 개념으로 cv2.multiply를 사용하거나 상수를 곱함
multiplied = cv2.multiply(img1, np.array([1.5])) 

# (4) 나눗셈: 이미지가 어두워지고 대비가 낮아짐
divided = cv2.divide(img1, np.array([2.0]))

# 3. 결과 시각화
titles = ['Original', 'Addition (+50)', 'Subtraction (-50)', 'Multiplication (x1.5)', 'Division (/2)']
images = [img1, added, subtracted, multiplied, divided]

plt.figure(figsize=(15, 8))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    # OpenCV는 BGR 순서이므로 시각화를 위해 RGB로 변환
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()