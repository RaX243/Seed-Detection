import ctypes
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(script_dir, "lib", "libwrapper.so")

print(f"脚本目录: {script_dir}")
if not os.path.exists(lib_path):
    print(f"错误: {lib_path} 不存在，请确保动态库文件存在于 lib 目录下！")
    exit(1)
libwrapper = ctypes.CDLL(lib_path)



# 开发日志:
# 2026-02-21: 
#   libwrapper.so动态文件是 ARM64 架构的，适用于树莓派4B等设备。
#   本机操作是不可以的，需要相应的开发环境和设备。
#   信息:
#   gouixn@fedora:~/document/stduent/Seed-Detection/pynirsdk_raspberry_python$ file lib/libwrapper.so
#   lib/libwrapper.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, BuildID[sha1]=ee3e752217886c49b3aab678b423c343f6773fe6, not stripped



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