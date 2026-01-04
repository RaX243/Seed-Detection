import ctypes
import wrapper
import matplotlib.pyplot as plt
import numpy as np

# 900-1700nm光谱模组是228个数据点
PIXEL_NUM = 228 

print("\n");

# 调用 dlpConnect() 函数
result = wrapper.dlpConnect()
print("连接设备数量:{}\n".format(result))

# 调用 dlpOpenByUsb() 函数 
index = 0  # USB 设备索引
result = wrapper.dlpOpenByUsb(index)
if(result>=0):
    print("hid usb 打开成功!\n")
else:
    print("hid usb 打开失败!\n")


# 创建用于存储波长数据的缓冲区
wls_buf = (ctypes.c_double * PIXEL_NUM)()


# 调用 dlpGetWavelengths() 函数
result = wrapper.dlpGetWavelengths(wls_buf, PIXEL_NUM)
if(result>=0):
    for i in range(5):
        wavelength = wls_buf[i]
        print("{:.2f}".format(wavelength)) #打印前5个波长，保留两位小数
else:
    print("获取波长失败!\n")

print("\n") # 波长和强度换行


# 创建用于存储强度数据的缓冲区
intensities_buf = (ctypes.c_int * PIXEL_NUM)()

# 调用 dlpGetIntensities() 函数
activeIndex = 0  # 激活的索引
result = wrapper.dlpGetIntensities(activeIndex, intensities_buf, PIXEL_NUM)
if(result>=0):
    # 取前5个强度并转换为普通列表
    intensities = [int(intensities_buf[i]) for i in range(5)]
    # 打印前5个波长和对应强度值
    for i, val in enumerate(intensities):
        print("{:.2f}: {}".format(wls_buf[i], val)) # 打印前5个波长和对应强度值

    plt.figure(figsize=(10, 5))

    x_positions = np.arange(1, 6)

    bars = plt.bar(x_positions, intensities, color='skyblue', edgecolor='black')

    for bar, val in zip(bars, intensities):
        height = val
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val}', ha='center', va='bottom', fontsize=10)

    plt.title('前5个波长对应的强度值', fontsize=14, fontweight='bold')
    plt.xlabel('数据点序号', fontsize=12)
    plt.ylabel('强度值', fontsize=12)

    plt.xticks(x_positions, ['1', '2', '3', '4', '5'])

    y_max = max(intensities) * 1.1  # 最大值留10%空间
    plt.ylim(0, y_max)

    plt.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()
else:
    print("获取强度失败!\n")
