import ctypes
import wrapper

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
    for i in range(5):
        intensity = intensities_buf[i]
        print("{:.2f}: {}".format(wavelength, intensity)) #打印前5个波长和对应强度值
else:
    print("获取强度失败!\n")
