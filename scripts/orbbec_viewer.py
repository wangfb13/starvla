import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0)) # 禁止生成core dump文件

import pyorbbecsdk as ob
from pyorbbecsdk import *
import cv2
import numpy as np
import time
from datetime import datetime

class OrbbecColorCameraViewer:
    def __init__(self):
        self.context = None
        self.devices = []
        self.pipelines = []
        self.windows_created = False
        self.running = False
        
    def list_all_devices(self):
        """列出所有可用设备"""
        self.context = ob.Context()
        device_list = self.context.query_devices()
        print(f"\n找到 {device_list.get_count()} 个奥比中科设备:")
        
        self.devices = []
        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            device_info = device.get_device_info()
            info = {
                'index': i,
                'device': device,
                'name': device_info.get_name(),
                'serial': device_info.get_serial_number(),
                'pid': device_info.get_pid(),
                'vid': device_info.get_vid(),
            }
            self.devices.append(info)
            print(f"{i}: {info['name']} (序列号: {info['serial']})")
        
        return self.devices
    
    def get_stream_profile_by_resolution(self, profile_list, target_width, target_height, target_format=None):
        """根据分辨率获取流配置"""
        if profile_list is None:
            return None
            
        for i in range(profile_list.get_count()):
            profile = profile_list.get_stream_profile_by_index(i)
            if profile.is_video_stream_profile():
                video_profile = profile.as_video_stream_profile()
                width = video_profile.get_width()
                height = video_profile.get_height()
                format_val = video_profile.get_format() if hasattr(video_profile, 'get_format') else None
                
                # 检查分辨率匹配
                if width == target_width and height == target_height:
                    # 如果指定了格式，检查格式匹配
                    if target_format is not None and format_val is not None:
                        if format_val == target_format:
                            return video_profile
                    else:
                        return video_profile
        
        print(f"未找到 {target_width}x{target_height} 分辨率的配置")
        return None
    
    def setup_pipeline_for_device(self, device, width=640, height=480):
        """为设备设置管道并配置640x480分辨率"""
        try:
            pipeline = Pipeline(device)
            
            # 配置流配置
            config = Config()
            
            # 设置彩色流 640x480
            color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profile_list:
                # 优先尝试RGB格式
                color_profile = self.get_stream_profile_by_resolution(
                    color_profile_list, width, height, OBFormat.RGB
                )
                if not color_profile:
                    # 如果RGB格式不可用，尝试YUYV格式
                    color_profile = self.get_stream_profile_by_resolution(
                        color_profile_list, width, height, OBFormat.YUYV
                    )
                if not color_profile:
                    # 如果特定格式不可用，只按分辨率查找
                    color_profile = self.get_stream_profile_by_resolution(
                        color_profile_list, width, height
                    )
                
                if color_profile:
                    config.enable_stream(color_profile)
                    format_name = self.get_format_name(color_profile.get_format())
                    print(f"  彩色: 已启用 {width}x{height} {format_name}")
                else:
                    print(f"  彩色: 未找到 {width}x{height} 的可用配置")
                    return None
            else:
                print(f"  彩色: 无法获取彩色传感器配置列表")
                return None
            
            # 启动管道
            pipeline.start(config)
            return pipeline
            
        except Exception as e:
            print(f"设置管道失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_format_name(self, format_val):
        """获取格式名称"""
        format_names = {
            OBFormat.RGB: "RGB",
            OBFormat.YUYV: "YUYV",
            OBFormat.MJPG: "MJPG",
            OBFormat.UYVY: "UYVY",
            OBFormat.NV21: "NV21",
            OBFormat.NV12: "NV12",
        }
        return format_names.get(format_val, str(format_val))
    
    def color_frame_to_image(self, color_frame):
        """将彩色帧转换为OpenCV图像"""
        if color_frame is None:
            return None
            
        width = color_frame.get_width()
        height = color_frame.get_height()
        
        # 根据格式转换
        format_val = color_frame.get_format()
        
        if format_val == OBFormat.RGB:
            # RGB格式
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_data = color_data.reshape((height, width, 3))
            # RGB转BGR用于OpenCV显示
            color_data = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
            return color_data
        elif format_val == OBFormat.YUYV:
            # YUYV格式
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_data = color_data.reshape((height, width, 2))
            color_data = cv2.cvtColor(color_data, cv2.COLOR_YUV2BGR_YUYV)
            return color_data
        elif format_val == OBFormat.MJPG:
            # MJPG格式
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_data = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            return color_data
        else:
            print(f"不支持的彩色格式: {format_val}")
            # 尝试通用处理
            try:
                color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                if len(color_data) == height * width * 3:
                    color_data = color_data.reshape((height, width, 3))
                    return color_data
            except:
                pass
            return None
    
    def create_windows(self, device_count):
        """为每个设备创建显示窗口"""
        for i in range(device_count):
            window_name = f'Device {i} - {self.devices[i]["name"]}'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
        self.windows_created = True
    
    def display_frames(self):
        """显示所有设备的彩色帧"""
        while self.running:
            for i, pipeline in enumerate(self.pipelines):
                if pipeline is None:
                    continue
                    
                try:
                    # 获取帧集
                    frames = pipeline.wait_for_frames(100)
                    if frames is None:
                        continue
                    
                    color_frame = frames.get_color_frame()
                    
                    # 处理彩色帧
                    color_image = None
                    if color_frame:
                        color_image = self.color_frame_to_image(color_frame)
                    
                    # 显示图像
                    if color_image is not None:
                        # 添加设备信息
                        device_info = self.devices[i]
                        window_name = f'Device {i} - {device_info["name"]}'
                        
                        # 在图像上添加信息文本
                        cv2.putText(color_image, f"Device {i}: {device_info['name']}", 
                                   (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(color_image, f"Serial: {device_info['serial']}", 
                                   (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
                        cv2.putText(color_image, f"Resolution: {color_image.shape[1]}x{color_image.shape[0]}", 
                                   (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                        
                        # # 显示帧率信息
                        # cv2.putText(color_image, datetime.now().strftime("%H:%M:%S"), 
                        #            (color_image.shape[1] - 150, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow(window_name, color_image)
                    
                except Exception as e:
                    print(f"设备 {i} 显示帧时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                self.running = False
                break
            elif key == ord('s'):  # 's' 保存当前帧
                self.save_current_frames()
            elif key == ord(' '):  # 空格键暂停/继续
                self.toggle_pause()
    
    def save_current_frames(self):
        """保存所有设备的当前帧到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, pipeline in enumerate(self.pipelines):
            if pipeline is None:
                continue
                
            try:
                frames = pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = self.color_frame_to_image(color_frame)
                    if color_image is not None:
                        device_info = self.devices[i]
                        filename = f"device_{i}_{device_info['serial']}_color_{timestamp}.png"
                        cv2.imwrite(filename, color_image)
                        print(f"保存设备 {i} 彩色图像: {filename}")
            except Exception as e:
                print(f"保存设备 {i} 图像时出错: {e}")
    
    def toggle_pause(self):
        """暂停/继续显示"""
        # 简单的暂停功能，通过等待特定按键实现
        print("暂停显示，按任意键继续...")
        cv2.waitKey(0)
        print("继续显示")
    
    def start_display(self):
        """开始显示所有相机"""
        if not self.devices:
            print("没有找到设备")
            return
        
        print(f"\n正在设置 {len(self.devices)} 个设备的管道...")
        
        # 为每个设备设置管道
        self.pipelines = []
        for i, device_info in enumerate(self.devices):
            print(f"\n设置设备 {i}: {device_info['name']}")
            pipeline = self.setup_pipeline_for_device(device_info['device'])
            if pipeline:
                self.pipelines.append(pipeline)
                print(f"✅ 设备 {i} 管道设置成功")
            else:
                self.pipelines.append(None)
                print(f"❌ 设备 {i} 管道设置失败")
        
        # 检查是否有成功设置的管道
        successful_pipelines = [p for p in self.pipelines if p is not None]
        if not successful_pipelines:
            print("没有成功设置任何管道")
            return
        
        print(f"\n成功设置 {len(successful_pipelines)} 个管道")
        print("开始显示彩色图像...")
        print("控制指令:")
        print("  按 'q' 或 ESC 退出")
        print("  按 's' 保存所有相机当前帧")
        print("  按 空格键 暂停/继续")
        
        # 创建显示窗口
        self.create_windows(len(self.devices))
        
        # 开始显示
        self.running = True
        try:
            self.display_frames()
        except KeyboardInterrupt:
            print("\n用户中断")
    
    def stop(self):
        """停止所有管道并清理资源"""
        self.running = False
        time.sleep(0.1)  # 给显示线程时间退出
        
        print("\n正在停止管道...") 
        for i, pipeline in enumerate(self.pipelines):
            if pipeline:
                try:
                    pipeline.stop()
                    print(f"设备 {i} 已停止")
                except Exception as e:
                    print(f"停止设备 {i} 时出错: {e}")
        
        cv2.destroyAllWindows()
        print("显示已关闭")

def main():
    """主函数"""
    viewer = OrbbecColorCameraViewer()
    
    try:
        # 列出所有设备
        devices = viewer.list_all_devices()
        if not devices:
            print("没有找到奥比中科设备，请检查连接")
            return
        
        # 开始显示
        viewer.start_display()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viewer.stop()

if __name__ == "__main__":
    main()