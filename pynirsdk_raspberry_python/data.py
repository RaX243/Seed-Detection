import ctypes
import wrapper


PIXEL_NUM = 228

# 主要功能函数
def acquire_and_plot_spectrum(device_index=0, plot_show=True, plot_save=None):
    """
    采集光谱数据并绘制图表
    
    参数:
        device_index (int): 设备索引， 默认为0
        plot_show (bool): 是否显示图表， 默认为True
        plot_save (str): 可选，保存图表的文件路径，如 "spectrum.png"
    
    返回:
        dict: 包含以下键的字典
            - 'success' (bool): 操作是否成功, 当不更改的时候默认是False, 在外部调用即可检查状态
            - 'wavelengths' (list): 波长数据列表 (nm)
            - 'intensities' (list): 强度数据列表
            - 'message' (str): 状态信息
    """

    result = {
        'success': False,
        'wavelengths': [],
        'intensities': [],
        'message': ''
    }
    
    # 1. 连接设备
    connect_result = wrapper.dlpConnect()
    if connect_result <= 0:
        result['message'] = "Error: 设备数量异常"
        print(result['message'])
        return result
    print(f"successfully connected to {connect_result} device(s).\n")
    
    # 2. 打开设备
    open_result = wrapper.dlpOpenByUsb(device_index)
    if open_result < 0:
        result['message'] = "Error: HID USB 失败"
        print(result['message'])
        return result
    print("HID USB 打开成功!\n")
    
    # 3. 获取波长数据
    wls_buf = (ctypes.c_double * PIXEL_NUM)()
    wls_result = wrapper.dlpGetWavelengths(wls_buf, PIXEL_NUM)
    if wls_result < 0:
        result['message'] = "获取波长失败!"
        print(result['message'])
        return result
    
    wavelengths = [float(wls_buf[i]) for i in range(PIXEL_NUM)]
    print("前5个波长:")
    for i in range(5):
        print(f"  {wavelengths[i]:.2f}")
    print()
    
    # 4. 获取强度数据
    intensities_buf = (ctypes.c_int * PIXEL_NUM)()
    activeIndex = 0
    int_result = wrapper.dlpGetIntensities(activeIndex, intensities_buf, PIXEL_NUM)
    if int_result < 0:
        result['message'] = "获取强度失败!"
        print(result['message'])
        return result
    
    intensities = [int(intensities_buf[i]) for i in range(PIXEL_NUM)]
    result['wavelengths'] = wavelengths
    result['intensities'] = intensities
    result['success'] = True
    result['message'] = "数据采集成功"
    return result


if __name__ == "__main__":
    data = acquire_and_plot_spectrum()
    if data['success']:
        print("数据采集成功，波长和强度已更新")
        print("前5个波长和强度:")
        print("波长 (nm) | 强度")
        for i in range(5):
            print(f"{data['wavelengths'][i]:.2f} | {data['intensities'][i]}")
    else:
        print("数据采集失败:", data['message'])

