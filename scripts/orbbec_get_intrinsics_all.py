import pyorbbecsdk as ob
from pyorbbecsdk import *
import json
import os
from datetime import datetime
import argparse

class OrbbecCameraCalibration:
    def __init__(self):
        self.pipeline = None
        
    def list_all_devices(self):
        """列出所有可用设备"""
        context = ob.Context()
        device_list = context.query_devices()
        print(f"\n找到 {device_list.get_count()} 个奥比设备:")
        
        devices_info = []
        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            device_info = device.get_device_info()
            info = {
                'index': i,
                'name': device_info.get_name(),
                'serial': device_info.get_serial_number(),
                'pid': device_info.get_pid(),
                'vid': device_info.get_vid(),
                'firmware_version': device_info.get_firmware_version() if hasattr(device_info, 'get_firmware_version') else 'unknown'
            }
            devices_info.append(info)
            print(f"{i}: {info['name']} (序列号: {info['serial']}, 固件版本: {info['firmware_version']})")
        
        return devices_info
    
    def find_device_by_serial(self, target_serial):
        """根据序列号查找设备"""
        context = ob.Context()
        device_list = context.query_devices()
        if device_list.get_count() == 0:
            print("未找到任何奥比设备")
            return None
            
        for i in range(device_list.get_count()):
            device = device_list.get_device_by_index(i)
            device_info = device.get_device_info()
            serial_number = device_info.get_serial_number()
            
            if serial_number == target_serial:
                print(f"找到目标设备: {serial_number}")
                return device
                
        print(f"未找到序列号为 {target_serial} 的设备")
        return None
    
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
                
                # 检查分辨率匹配
                if width == target_width and height == target_height:
                    # 如果指定了格式，检查格式匹配
                    if target_format is not None:
                        if hasattr(video_profile, 'get_format'):
                            format_val = video_profile.get_format()
                            if format_val == target_format:
                                return video_profile
                        else:
                            # 如果不支持格式检查，直接返回
                            return video_profile
                    else:
                        return video_profile
        
        print(f"未找到 {target_width}x{target_height} 分辨率的配置")
        return None
    
    def intrinsics_to_dict(self, intrinsics, sensor_type, resolution):
        """将内参对象转换为字典"""
        return {
            'sensor_type': str(sensor_type),
            'resolution': f"{resolution[0]}x{resolution[1]}",
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.cx,
            'cy': intrinsics.cy,
            'pixel_format': str(intrinsics.pixel_format) if hasattr(intrinsics, 'pixel_format') else 'unknown'
        }
    
    def distortion_to_dict(self, distortion, sensor_type, resolution):
        """将畸变参数对象转换为字典"""
        return {
            'sensor_type': str(sensor_type),
            'resolution': f"{resolution[0]}x{resolution[1]}",
            'model': str(distortion.model) if hasattr(distortion, 'model') else 'unknown',
            'coefficients': {
                'k1': distortion.k1 if hasattr(distortion, 'k1') else 0,
                'k2': distortion.k2 if hasattr(distortion, 'k2') else 0,
                'p1': distortion.p1 if hasattr(distortion, 'p1') else 0,
                'p2': distortion.p2 if hasattr(distortion, 'p2') else 0,
                'k3': distortion.k3 if hasattr(distortion, 'k3') else 0,
                'k4': distortion.k4 if hasattr(distortion, 'k4') else 0,
                'k5': distortion.k5 if hasattr(distortion, 'k5') else 0,
                'k6': distortion.k6 if hasattr(distortion, 'k6') else 0,
            }
        }
    
    def extrinsic_to_dict(self, extrinsic, from_resolution, to_resolution):
        """将外参对象转换为字典"""
        return {
            'from_resolution': f"{from_resolution[0]}x{from_resolution[1]}",
            'to_resolution': f"{to_resolution[0]}x{to_resolution[1]}",
            'rotation': [
                [float(extrinsic.rot[0][0]), float(extrinsic.rot[0][1]), float(extrinsic.rot[0][2])],
                [float(extrinsic.rot[1][0]), float(extrinsic.rot[1][1]), float(extrinsic.rot[1][2])],
                [float(extrinsic.rot[2][0]), float(extrinsic.rot[2][1]), float(extrinsic.rot[2][2])]
            ],
            'translation': [
                float(extrinsic.transform[0]),
                float(extrinsic.transform[1]),
                float(extrinsic.transform[2])
            ]
        }
    
    def save_intrinsics_data(self, data, serial_number, output_dir="intrinsics_data"):
        """保存校准数据到JSON文件"""
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建保存目录: {os.path.abspath(output_dir)}")
            
        # 生成文件名
        filename = f"orbbec_intrinsics_{serial_number}_all_resolutions.json"
        filepath = os.path.join(output_dir, filename)
        
        # 直接覆盖已存在的文件（不备份）
        if os.path.exists(filepath):
            print(f"已存在同名文件，将直接覆盖: {filename}")
        
        # 保存数据
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"校准数据已保存到: {filepath}")
        return filepath
    
    def get_available_resolutions(self, device, sensor_type):
        """获取设备支持的分辨率列表"""
        try:
            pipeline = Pipeline(device)
            profile_list = pipeline.get_stream_profile_list(sensor_type)
            resolutions = set()
            
            if profile_list:
                for i in range(profile_list.get_count()):
                    profile = profile_list.get_stream_profile_by_index(i)
                    if profile.is_video_stream_profile():
                        video_profile = profile.as_video_stream_profile()
                        resolutions.add((video_profile.get_width(), video_profile.get_height()))
            
            pipeline.stop()
            return sorted(resolutions, key=lambda x: (x[0], x[1]))
        except:
            return []
    
    def get_color_intrinsics_for_resolution(self, device, color_resolution):
        """获取彩色相机指定分辨率的内参"""
        try:
            self.pipeline = Pipeline(device)
            
            # 获取彩色传感器的配置列表
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            
            if color_profile_list is None:
                print(f"无法获取彩色传感器配置列表")
                return None
            
            # 获取指定分辨率的配置
            color_profile = self.get_stream_profile_by_resolution(
                color_profile_list, color_resolution[0], color_resolution[1]
            )
            
            if color_profile is None:
                print(f"无法获取彩色传感器 {color_resolution[0]}x{color_resolution[1]} 配置")
                return None
            
            # 获取内参
            color_intrinsics = color_profile.get_intrinsic()
            
            # 获取畸变参数
            color_distortion = color_profile.get_distortion()
            
            # 构建该分辨率的彩色相机数据
            color_data = {
                'resolution': f"{color_resolution[0]}x{color_resolution[1]}",
                'intrinsics': self.intrinsics_to_dict(color_intrinsics, OBSensorType.COLOR_SENSOR, color_resolution),
                'distortion': self.distortion_to_dict(color_distortion, OBSensorType.COLOR_SENSOR, color_resolution)
            }
            
            return color_data
            
        except Exception as e:
            print(f"获取彩色相机 {color_resolution[0]}x{color_resolution[1]} 内参过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_depth_intrinsics_for_resolution(self, device, depth_resolution):
        """获取深度相机指定分辨率的内参"""
        try:
            self.pipeline = Pipeline(device)
            
            # 获取深度传感器的配置列表
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            
            if depth_profile_list is None:
                print(f"无法获取深度传感器配置列表")
                return None
            
            # 获取指定分辨率的配置
            depth_profile = self.get_stream_profile_by_resolution(
                depth_profile_list, depth_resolution[0], depth_resolution[1]
            )
            
            if depth_profile is None:
                print(f"无法获取深度传感器 {depth_resolution[0]}x{depth_resolution[1]} 配置")
                return None
            
            # 获取内参
            depth_intrinsics = depth_profile.get_intrinsic()
            
            # 获取畸变参数
            depth_distortion = depth_profile.get_distortion()
            
            # 构建该分辨率的深度相机数据
            depth_data = {
                'resolution': f"{depth_resolution[0]}x{depth_resolution[1]}",
                'intrinsics': self.intrinsics_to_dict(depth_intrinsics, OBSensorType.DEPTH_SENSOR, depth_resolution),
                'distortion': self.distortion_to_dict(depth_distortion, OBSensorType.DEPTH_SENSOR, depth_resolution)
            }
            
            return depth_data
            
        except Exception as e:
            print(f"获取深度相机 {depth_resolution[0]}x{depth_resolution[1]} 内参过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_extrinsic_for_resolution_pair(self, device, color_resolution, depth_resolution):
        """获取指定分辨率对的外参"""
        try:
            self.pipeline = Pipeline(device)
            
            # 获取彩色和深度传感器的配置列表
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            
            if color_profile_list is None or depth_profile_list is None:
                return None
            
            # 获取指定分辨率的配置
            color_profile = self.get_stream_profile_by_resolution(
                color_profile_list, color_resolution[0], color_resolution[1]
            )
            depth_profile = self.get_stream_profile_by_resolution(
                depth_profile_list, depth_resolution[0], depth_resolution[1]
            )
            
            if color_profile is None or depth_profile is None:
                return None
            
            # 获取外参（深度到彩色的变换）
            extrinsic = depth_profile.get_extrinsic_to(color_profile)
            
            extrinsic_data = {
                'from_resolution': f"{depth_resolution[0]}x{depth_resolution[1]}",
                'to_resolution': f"{color_resolution[0]}x{color_resolution[1]}",
                'extrinsic': self.extrinsic_to_dict(extrinsic, depth_resolution, color_resolution)
            }
            
            return extrinsic_data
            
        except Exception as e:
            print(f"获取外参过程中出错: {e}")
            return None
    
    def get_calibration(self, target_serial, save_dir="intrinsics_data"):
        """获取指定序列号设备的所有分辨率校准数据
        
        Args:
            target_serial (str): 目标设备的序列号
            save_dir (str): 保存目录路径，默认为 'intrinsics_data'
        """
        # 查找设备
        device = self.find_device_by_serial(target_serial)
        if device is None:
            return False
            
        try:
            # 获取设备信息
            device_info = device.get_device_info()
            device_name = device_info.get_name()
            device_pid = device_info.get_pid()
            device_vid = device_info.get_vid()
            firmware_version = device_info.get_firmware_version() if hasattr(device_info, 'get_firmware_version') else 'unknown'
            
            # 获取所有可用的分辨率
            available_color_res = self.get_available_resolutions(device, OBSensorType.COLOR_SENSOR)
            available_depth_res = self.get_available_resolutions(device, OBSensorType.DEPTH_SENSOR)
            
            print(f"\n设备支持的分辨率:")
            print(f"  彩色: {available_color_res}")
            print(f"  深度: {available_depth_res}")
            
            # 收集彩色相机所有分辨率的内参
            print(f"\n正在获取彩色相机所有分辨率的内参...")
            color_calibrations = []
            for color_res in available_color_res:
                print(f"  获取彩色 {color_res[0]}x{color_res[1]}...")
                color_data = self.get_color_intrinsics_for_resolution(device, color_res)
                if color_data:
                    color_calibrations.append(color_data)
                #     print(f"    ✅ 成功")
                # else:
                #     print(f"    ❌ 失败")
            
            # 收集深度相机所有分辨率的内参
            print(f"\n正在获取深度相机所有分辨率的内参...")
            depth_calibrations = []
            for depth_res in available_depth_res:
                print(f"  获取深度 {depth_res[0]}x{depth_res[1]}...")
                depth_data = self.get_depth_intrinsics_for_resolution(device, depth_res)
                if depth_data:
                    depth_calibrations.append(depth_data)
                #     print(f"    ✅ 成功")
                # else:
                #     print(f"    ❌ 失败")
            
            # 获取默认分辨率的外参（使用第一个可用的分辨率）
            extrinsic_data = None
            if available_color_res and available_depth_res:
                default_color_res = available_color_res[0]
                default_depth_res = available_depth_res[0]
                print(f"\n获取外参 (深度 {default_depth_res[0]}x{default_depth_res[1]} -> 彩色 {default_color_res[0]}x{default_color_res[1]})...")
                extrinsic_data = self.get_extrinsic_for_resolution_pair(device, default_color_res, default_depth_res)
                # if extrinsic_data:
                #     print(f"  ✅ 成功")
                # else:
                #     print(f"  ❌ 失败")
            
            # 构建完整的校准数据
            intrinsics_data = {
                'device_info': {
                    'name': device_name,
                    'serial_number': target_serial,
                    'pid': device_pid,
                    'vid': device_vid,
                    'firmware_version': firmware_version,
                    'calibration_date': datetime.now().isoformat(),
                    'save_directory': os.path.abspath(save_dir),
                    'available_resolutions': {
                        'color': [f"{w}x{h}" for w, h in available_color_res],
                        'depth': [f"{w}x{h}" for w, h in available_depth_res]
                    },
                    'successful_calibrations': {
                        'color': len(color_calibrations),
                        'depth': len(depth_calibrations)
                    },
                    'depth_scale': 1.0,
                    'depth_unit': 'millimeter',
                    'depth_conversion_formula': 'depth_in_millimeter = depth_raw * depth_scale'
                },
                'color_calibrations': color_calibrations,
                'depth_calibrations': depth_calibrations
            }
            
            # 添加外参数据（如果获取成功）
            if extrinsic_data:
                intrinsics_data['extrinsic'] = extrinsic_data
            
            # 保存数据
            saved_path = self.save_intrinsics_data(intrinsics_data, target_serial, save_dir)
            
            # 打印结果摘要
            print("\n" + "="*60)
            print("多分辨率相机校准数据获取完成!")
            print("="*60)
            
            print(f"\n设备信息:")
            print(f"  名称: {device_name}")
            print(f"  序列号: {target_serial}")
            print(f"  固件版本: {firmware_version}")
            print(f"  成功获取的校准数据:")
            print(f"    彩色: {len(color_calibrations)} 个分辨率")
            print(f"    深度: {len(depth_calibrations)} 个分辨率")
            print(f"  保存目录: {os.path.abspath(save_dir)}")
            
            print(f"\n彩色相机内参分辨率:")
            for color_data in color_calibrations:
                res = color_data['resolution']
                intrinsics = color_data['intrinsics']
                print(f"  {res}: fx={intrinsics['fx']:.3f}, fy={intrinsics['fy']:.3f}, cx={intrinsics['cx']:.3f}, cy={intrinsics['cy']:.3f}")
            
            print(f"\n深度相机内参分辨率:")
            for depth_data in depth_calibrations:
                res = depth_data['resolution']
                intrinsics = depth_data['intrinsics']
                print(f"  {res}: fx={intrinsics['fx']:.3f}, fy={intrinsics['fy']:.3f}, cx={intrinsics['cx']:.3f}, cy={intrinsics['cy']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"获取内参过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理资源
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except:
                    pass

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='获取奥比相机所有分辨率内参')
    parser.add_argument('--save-dir', '-d', type=str, default='intrinsics_data',
                       help='保存内参文件的目录路径，默认为 intrinsics_data')
    parser.add_argument('--list-only', '-l', action='store_true',
                       help='仅列出设备，不获取内参')
    parser.add_argument('--serial', '-s', type=str,
                       help='指定单个设备序列号，只获取该设备的内参')
    
    args = parser.parse_args()
    
    # 创建校准器实例
    calibrator = OrbbecCameraCalibration()
    
    # 列出所有设备
    devices = calibrator.list_all_devices()
    
    if not devices:
        print("没有找到任何设备，请检查连接")
        return
    
    if args.list_only:
        # 仅列出设备
        return
    
    # 确定要处理的设备列表
    target_devices = []
    
    if args.serial:
        # 使用命令行指定的单个序列号
        target_serial = args.serial
        if any(device['serial'] == target_serial for device in devices):
            target_devices = [target_serial]
            print(f"\n将获取指定设备的内参: {target_serial}")
        else:
            print(f"错误: 序列号 {target_serial} 的设备未找到")
            return
    else:
        # 默认获取所有设备的内参（直接确认，不询问）
        target_devices = [device['serial'] for device in devices]
        print(f"\n将自动获取所有 {len(target_devices)} 个设备的内参")
        print(f"将处理以下设备:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']} (序列号: {device['serial']})")
    
    print(f"\n保存目录: {os.path.abspath(args.save_dir)}")
    
    # 为每个设备获取内参
    success_count = 0
    failed_count = 0
    
    for i, serial in enumerate(target_devices):
        print(f"\n{'='*80}")
        print(f"正在处理设备 {i+1}/{len(target_devices)}: {serial}")
        print(f"{'='*80}")
        
        success = calibrator.get_calibration(serial, args.save_dir)
        
        if success:
            success_count += 1
            print(f"✅ 设备 {serial} 内参获取成功")
        else:
            failed_count += 1
            print(f"❌ 设备 {serial} 内参获取失败")
    
    # 打印最终结果
    print(f"\n{'='*80}")
    print("所有设备处理完成!")
    print(f"{'='*80}")
    print(f"成功: {success_count} 个设备")
    print(f"失败: {failed_count} 个设备")
    print(f"保存目录: {os.path.abspath(args.save_dir)}")
    
    if success_count > 0:
        print(f"\n✅ 操作成功完成! 共保存了 {success_count} 个设备的内参数据")
    else:
        print(f"\n❌ 所有设备的内参获取都失败了!")

# 使用示例
if __name__ == "__main__":
    main()