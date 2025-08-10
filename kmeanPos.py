import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_excel("beauty-data.xlsx")

# 2. Chọn cột để huấn luyện
X = df[['Price', 'Popularity_score']].copy()

# 3. Chuẩn hóa dữ liệu để huấn luyện
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Chạy KMeans trên dữ liệu chuẩn hóa
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# 5. Gán nhãn cụm vào dữ liệu gốc
df['Cluster_Label'] = kmeans.labels_

# 6. Lấy centroid về giá trị gốc
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

# 7. Vẽ biểu đồ
plt.figure(figsize=(8, 5))
colors = ['blue', 'orange', 'green']

for i in range(k):
    plt.scatter(
        df.loc[df['Cluster_Label'] == i, 'Price'],
        df.loc[df['Cluster_Label'] == i, 'Popularity_score'],
        s=20,
        c=colors[i],
        label=f'Cụm {i+1}'
    )

# Vẽ centroid
plt.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    c='black', s=100, marker='o'
)

# Ghi nhãn centroid
for i, (x, y) in enumerate(centers_original):
    plt.text(
        x, y,
        f"Cụm {i+1}\n({x:,.0f} VND, {y:.2f})",
        fontsize=9, ha='center', va='bottom',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    )

plt.title("Phân Cụm K-Means - Sản phẩm mỹ phẩm (Giá trị gốc)", fontsize=14)
plt.xlabel("Giá (VND)")
plt.ylabel("Độ nổi tiếng")
plt.legend()

# Định dạng trục X thành dạng VND
plt.ticklabel_format(style='plain', axis='x')  # bỏ scientific notation
plt.gca().get_xaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
)

plt.show()
