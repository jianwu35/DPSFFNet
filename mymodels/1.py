import pywt

# 获取所有支持的小波名称
wavelet_list = pywt.wavelist()

# 打印小波名称列表
print(wavelet_list)