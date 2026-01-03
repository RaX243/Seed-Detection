import ctypes

# 加载共享库
libwrapper = ctypes.CDLL("lib/libwrapper.so")

# ----定义函数原型----
# libwrapper.so是由c语言封装的动态库，python调用库函数时，需要先指定参数类型为ctypes。

# 连接光谱仪设备。返回值是已连接设备数量。
libwrapper.dlpConnect.argtypes = []
libwrapper.dlpConnect.restype = ctypes.c_int

# 打开已连接的光谱仪设备。参数是c_int型的配置索引；返回值<0打开失败，返回值>=0打开成功。
libwrapper.dlpOpenByUsb.argtypes = [ctypes.c_int]
libwrapper.dlpOpenByUsb.restype = ctypes.c_int

# 获取波长。参数是c_double型的指针（POINTER）和c_int型的波长个数；返回值<0打开失败，返回值>=0打开成功。
libwrapper.dlpGetWavelengths.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
libwrapper.dlpGetWavelengths.restype = ctypes.c_int

# 获取强度值。参数是c_int型的指针（POINTER）和c_int型的数据点数；返回值<0打开失败，返回值>=0打开成功。
libwrapper.dlpGetIntensities.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
libwrapper.dlpGetIntensities.restype = ctypes.c_int

# ----/定义函数原型----



# ----调用函数----
# 连接光谱仪设备。返回值是已连接设备数量。
def dlpConnect():
    return libwrapper.dlpConnect()

# 打开已连接的光谱仪设备。参数是c_int型的配置索引；返回值<0打开失败，返回值>=0打开成功。
def dlpOpenByUsb(index):
    return libwrapper.dlpOpenByUsb(index)

# 获取波长。参数是c_double型的指针（POINTER）和c_int型的波长个数；返回值<0打开失败，返回值>=0打开成功。
def dlpGetWavelengths(wls, wlsNum):
    return libwrapper.dlpGetWavelengths(wls, wlsNum)

# 获取强度值。参数是c_int型的指针（POINTER）和c_int型的数据点数；返回值<0打开失败，返回值>=0打开成功。
def dlpGetIntensities(activeIndex, intensities, wlsNum):
    return libwrapper.dlpGetIntensities(activeIndex, intensities, wlsNum)

# ----/调用函数----