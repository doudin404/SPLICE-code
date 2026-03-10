import pickle



# 读取 pkl 文件
with open('./export/save/1299.pkl', 'rb') as file:
    data = pickle.load(file)

# with open('./assets/snowman_decoded_mesh/chair_0.pkl', 'rb') as file:
#     data = pickle.load(file)


# 输出文件内容
# 16, 512
print(data)
print(data.shape)
