import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# def read_data(file_path):
#     """Đọc dữ liệu từ file Excel"""
#     df = pd.read_excel(file_path)
#     return df

# def preprocess_data(df):
#     """Xử lý dữ liệu: chọn cột số, điền giá trị thiếu, chuẩn hóa"""
#     numeric_cols = ['Price', 'Volume_ml', 'SPF', 'Popularity_score']
#     df[numeric_cols] = df[numeric_cols].fillna(0)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df[numeric_cols])
#     return X_scaled, df

# def kmeans_clustering(X_scaled, n_clusters=3):
#     """Chạy K-means và trả về model cùng nhãn cụm"""
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(X_scaled)
#     return kmeans

# def main():
#     # 1. Đọc dữ liệu
#     file_path = "beauty-data.xlsx"  # Thay bằng đường dẫn thực tế
#     df = read_data(file_path)

#     # 2. Tiền xử lý
#     X_scaled, df = preprocess_data(df)

#     # 3. Chạy K-means
#     kmeans = kmeans_clustering(X_scaled, n_clusters=3)
#     df['Cluster_Label'] = kmeans.labels_

#     # 4. In thông tin từng cụm
#     for cluster_id in sorted(df['Cluster_Label'].unique()):
#         print(f"\n=== Cụm {cluster_id} ===")
#         cluster_data = df[df['Cluster_Label'] == cluster_id]
#         print(cluster_data[['Product_Name', 'Price', 'Volume_ml', 'SPF', 'Brand', 'Popularity_score', 'Category']])

#     # 5. Xuất file kết quả
#     df.to_excel("clustered_data.xlsx", index=False)
#     print("\n✅ Đã lưu file kết quả: clustered_data.xlsx")

# if __name__ == "__main__":
#     main()



# 1. Đọc dữ liệu từ Excel
# Thay 'data.xlsx' bằng tên file thật của bạn
df = pd.read_excel("beauty-data.xlsx")

# 2. Chọn 2 cột để trực quan hóa (vd: Price và Rating)
X = df[['Price', 'Popularity_score']].copy()

# 3. Chuẩn hóa dữ liệu để tránh bias do đơn vị
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Áp dụng K-Means
k = 3  # Số cụm mong muốn
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# 5. Lấy kết quả
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 6. Vẽ biểu đồ giống mẫu
plt.figure(figsize=(6, 4))
colors = ['blue', 'orange', 'green']  # Màu cho từng cụm

for i in range(k):
    plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1],
                s=20, c=colors[i], label=f'Cụm {i+1}')

# Vẽ centroid
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='o')

plt.title("Phân Cụm K-Means - Sản phẩm mỹ phẩm", fontsize=14)
plt.xlabel("Giá (chuẩn hóa)")
plt.ylabel("Độ nổi tiếng(chuẩn hóa)")
plt.legend()
plt.show()
