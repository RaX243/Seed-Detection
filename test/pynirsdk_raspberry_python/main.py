import ctypes
import wrapper
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import font_manager
import os

# 设置中文字体
preferred = ['noto', 'wqy', 'simhei', 'msyh', 'msjh', 'arphic', 'ukai']
found_font = None
font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf') + font_manager.findSystemFonts(fontpaths=None, fontext='otf')
for fpath in font_paths:
    name = os.path.basename(fpath).lower()
    if any(p in name for p in preferred):
        try:
            prop = font_manager.FontProperties(fname=fpath)
            found_font = prop.get_name()
            break
        except Exception:
            continue
if found_font:
    matplotlib.rcParams['font.sans-serif'] = [found_font]
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PIXEL_NUM = 228

print()

result = wrapper.dlpConnect()
print("连接设备数量:{}\n".format(result))

index = 0
result = wrapper.dlpOpenByUsb(index)
if result >= 0:
    print("hid usb 打开成功!\n")
else:
    print("hid usb 打开失败!\n")

wls_buf = (ctypes.c_double * PIXEL_NUM)()
result = wrapper.dlpGetWavelengths(wls_buf, PIXEL_NUM)
if result >= 0:
    for i in range(5):
        print("{:.2f}".format(wls_buf[i]))
else:
    print("获取波长失败!\n")

print()

intensities_buf = (ctypes.c_int * PIXEL_NUM)()
activeIndex = 0
result = wrapper.dlpGetIntensities(activeIndex, intensities_buf, PIXEL_NUM)
if result >= 0:
    intensities = [int(intensities_buf[i]) for i in range(5)]
    wavelengths = [float(wls_buf[i]) for i in range(5)]
    for wl, val in zip(wavelengths, intensities):
        print("{:.2f}: {}".format(wl, val))

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, intensities, marker='o', linestyle='-', color='tab:blue')
    plt.title('前5个波长对应的强度值', fontsize=14, fontweight='bold')
    plt.xlabel('波长 (nm)', fontsize=12)
    plt.ylabel('强度值', fontsize=12)
    y_max = max(intensities) * 1.1
    plt.ylim(0, y_max)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
else:
    print("获取强度失败!\n")
