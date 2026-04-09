import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 연산을 위한 이미지 생성 (검은 바탕에 흰색 도형)
rect = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(rect, (50, 50), (250, 250), 255, -1)

circle = np.zeros((300, 300), dtype="uint8")
cv2.circle(circle, (150, 150), 120, 255, -1)

# 2. 논리 연산 수행
bit_and = cv2.bitwise_and(rect, circle)  # 교집합
bit_or = cv2.bitwise_or(rect, circle)    # 합집합
bit_xor = cv2.bitwise_xor(rect, circle)  # 배타적 논리합
bit_not = cv2.bitwise_not(rect)          # 반전

# 3. 시각화
titles = ['Rectangle', 'Circle', 'AND', 'OR', 'XOR', 'NOT (Rect)']
images = [rect, circle, bit_and, bit_or, bit_xor, bit_not]

plt.figure(figsize=(15, 5))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.show()