#!/usr/bin/env python
"""
震源机制解到矩张量转换程序
Focal Mechanism to Moment Tensor Converter
作者: MTUQ
用途: 将震源机制解(走向/倾角/滑动角)转换为地震矩张量
使用方法:
1. 交互式: python focal_to_mt.py
2. 命令行: python focal_to_mt.py --strike 225 --dip 45 --rake -90
3. 批处理: python focal_to_mt.py --input mechanisms.txt --output results.txt
"""
import sys
import os
import argparse
import numpy as np
from typing import Tuple, List, Dict, Optional
import json
# 检查并安装依赖（可选）
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("提示: matplotlib未安装，图形功能不可用")
class FocalMechanismToMomentTensor:
    """
    震源机制解到矩张量转换器
    """

    def __init__(self):
        """初始化转换器"""
        self.results = []

    def strike_dip_rake_to_mt(self, 
                              strike: float, 
                              dip: float, 
                              rake: float, 
                              magnitude: Optional[float] = None) -> Dict:
        """
        将震源机制解(走向、倾角、滑动角)转换为矩张量

        Parameters:
        -----------
        strike : float
            走向 (0-360度)
        dip : float
            倾角 (0-90度)
        rake : float
            滑动角 (-180-180度)
        magnitude : float, optional
            矩震级 (用于计算标量矩)

        Returns:
        --------
        result : dict
            包含矩张量分量和其他参数
        """
        # 输入验证
        self._validate_input(strike, dip, rake)

        # 转换为弧度
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        rake_rad = np.radians(rake)

        # 计算三角函数值
        sin_strike = np.sin(strike_rad)
        cos_strike = np.cos(strike_rad)
        sin_dip = np.sin(dip_rad)
        cos_dip = np.cos(dip_rad)
        sin_rake = np.sin(rake_rad)
        cos_rake = np.cos(rake_rad)
        sin_2strike = np.sin(2 * strike_rad)
        cos_2strike = np.cos(2 * strike_rad)
        sin_2dip = np.sin(2 * dip_rad)
        cos_2dip = np.cos(2 * dip_rad)

        # 计算归一化矩张量分量 (M0 = 1)
        # 使用Aki & Richards (2002)约定
        Mrr = sin_2dip * sin_rake
        Mtt = -sin_dip * cos_rake * sin_2strike - sin_2dip * sin_rake * sin_strike**2
        Mpp = sin_dip * cos_rake * sin_2strike - sin_2dip * sin_rake * cos_strike**2
        Mrt = -cos_dip * cos_rake * cos_strike - cos_2dip * sin_rake * sin_strike
        Mrp = cos_dip * cos_rake * sin_strike - cos_2dip * sin_rake * cos_strike
        Mtp = -sin_dip * cos_rake * cos_2strike - 0.5 * sin_2dip * sin_rake * sin_2strike

        # 如果提供了震级，计算实际矩张量
        if magnitude is not None:
            M0 = self.magnitude_to_moment(magnitude)
            Mrr *= M0
            Mtt *= M0
            Mpp *= M0
            Mrt *= M0
            Mrp *= M0
            Mtp *= M0
        else:
            M0 = 1.0

        # 构建结果字典
        result = {
            'input': {
                'strike': strike,
                'dip': dip,
                'rake': rake,
                'magnitude': magnitude
            },
            'moment_tensor': {
                'Mrr': Mrr,
                'Mtt': Mtt,
                'Mpp': Mpp,
                'Mrt': Mrt,
                'Mrp': Mrp,
                'Mtp': Mtp
            },
            'scalar_moment': M0,
            'fault_type': self._classify_fault_type(rake),
            'coordinate_system': 'USE (Up-South-East)',
            'convention': 'Aki & Richards (2002)'
        }

        # 计算主轴
        principal_axes = self._calculate_principal_axes(
            [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
        )
        result['principal_axes'] = principal_axes

        # 计算双力偶百分比
        result['double_couple_percentage'] = self._calculate_dc_percentage(
            principal_axes
        )

        self.results.append(result)
        return result

    def _validate_input(self, strike: float, dip: float, rake: float):
        """验证输入参数"""
        if not 0 <= strike <= 360:
            raise ValueError(f"走向必须在0-360度之间，当前值: {strike}")
        if not 0 <= dip <= 90:
            raise ValueError(f"倾角必须在0-90度之间，当前值: {dip}")
        if not -180 <= rake <= 180:
            raise ValueError(f"滑动角必须在-180到180度之间，当前值: {rake}")

    def _classify_fault_type(self, rake: float) -> str:
        """根据滑动角分类断层类型"""
        if -135 <= rake <= -45:
            return "正断层 (Normal)"
        elif -45 < rake < 45:
            return "走滑断层 (Strike-slip)"
        elif 45 <= rake <= 135:
            return "逆断层 (Reverse)"
        else:
            return "走滑断层 (Strike-slip)"

    def magnitude_to_moment(self, magnitude: float) -> float:
        """
        将矩震级转换为标量地震矩

        Hanks & Kanamori (1979): Mw = (2/3) * log10(M0) - 10.7
        因此: M0 = 10^(1.5 * Mw + 9.1) (单位: N·m)
        """
        return 10 ** (1.5 * magnitude + 9.1)

    def moment_to_magnitude(self, moment: float) -> float:
        """将标量地震矩转换为矩震级"""
        return (2.0 / 3.0) * (np.log10(moment) - 9.1)

    def _calculate_principal_axes(self, mt_components: List[float]) -> Dict:
        """计算主轴（T、B、P轴）"""
        # 构建矩张量矩阵
        Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = mt_components

        mt_matrix = np.array([
            [Mrr, Mrt, Mrp],
            [Mrt, Mtt, Mtp],
            [Mrp, Mtp, Mpp]
        ])

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(mt_matrix)

        # 排序（从大到小）
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # T轴（拉张轴）
        T = {
            'value': eigenvalues[0],
            'azimuth': self._vector_to_azimuth(eigenvectors[:, 0]),
            'plunge': self._vector_to_plunge(eigenvectors[:, 0])
        }

        # B轴（中间轴）
        B = {
            'value': eigenvalues[1],
            'azimuth': self._vector_to_azimuth(eigenvectors[:, 1]),
            'plunge': self._vector_to_plunge(eigenvectors[:, 1])
        }

        # P轴（压缩轴）
        P = {
            'value': eigenvalues[2],
            'azimuth': self._vector_to_azimuth(eigenvectors[:, 2]),
            'plunge': self._vector_to_plunge(eigenvectors[:, 2])
        }

        return {'T': T, 'B': B, 'P': P}

    def _vector_to_azimuth(self, vector: np.ndarray) -> float:
        """将向量转换为方位角"""
        azimuth = np.degrees(np.arctan2(vector[2], vector[1]))
        if azimuth < 0:
            azimuth += 360
        return azimuth

    def _vector_to_plunge(self, vector: np.ndarray) -> float:
        """将向量转换为倾伏角"""
        return np.degrees(np.arcsin(-vector[0]))

    def _calculate_dc_percentage(self, principal_axes: Dict) -> float:
        """计算双力偶成分百分比"""
        T = principal_axes['T']['value']
        B = principal_axes['B']['value']
        P = principal_axes['P']['value']

        # Frohlich (1992) 方法
        epsilon = -B / max(abs(T), abs(P))
        dc_percentage = (1 - 2 * abs(epsilon)) * 100

        return dc_percentage

    def print_results(self, result: Dict, detailed: bool = True):
        """打印结果"""
        print("\n" + "="*60)
        print("震源机制解到矩张量转换结果")
        print("="*60)

        # 输入参数
        inp = result['input']
        print(f"\n输入参数:")
        print(f"  走向 (Strike): {inp['strike']:.1f}°")
        print(f"  倾角 (Dip): {inp['dip']:.1f}°")
        print(f"  滑动角 (Rake): {inp['rake']:.1f}°")
        if inp['magnitude']:
            print(f"  震级 (Magnitude): Mw {inp['magnitude']:.2f}")

        # 断层类型
        print(f"\n断层类型: {result['fault_type']}")

        # 矩张量分量
        mt = result['moment_tensor']
        print(f"\n矩张量分量 ({result['coordinate_system']}):")

        if inp['magnitude']:
            # 科学计数法显示
            print(f"  Mrr = {mt['Mrr']:.3e} N·m")
            print(f"  Mtt = {mt['Mtt']:.3e} N·m")
            print(f"  Mpp = {mt['Mpp']:.3e} N·m")
            print(f"  Mrt = {mt['Mrt']:.3e} N·m")
            print(f"  Mrp = {mt['Mrp']:.3e} N·m")
            print(f"  Mtp = {mt['Mtp']:.3e} N·m")
            print(f"\n标量地震矩 M0 = {result['scalar_moment']:.3e} N·m")
        else:
            # 归一化值
            print(f"  Mrr = {mt['Mrr']:+.4f}")
            print(f"  Mtt = {mt['Mtt']:+.4f}")
            print(f"  Mpp = {mt['Mpp']:+.4f}")
            print(f"  Mrt = {mt['Mrt']:+.4f}")
            print(f"  Mrp = {mt['Mrp']:+.4f}")
            print(f"  Mtp = {mt['Mtp']:+.4f}")
            print(f"\n（归一化值，M0 = 1）")

        if detailed:
            # 主轴
            axes = result['principal_axes']
            print(f"\n主轴分析:")
            print(f"  T轴 (拉张): 方位角={axes['T']['azimuth']:.1f}°, "
                  f"倾伏角={axes['T']['plunge']:.1f}°")
            print(f"  B轴 (中间): 方位角={axes['B']['azimuth']:.1f}°, "
                  f"倾伏角={axes['B']['plunge']:.1f}°")
            print(f"  P轴 (压缩): 方位角={axes['P']['azimuth']:.1f}°, "
                  f"倾伏角={axes['P']['plunge']:.1f}°")

            # 双力偶百分比
            print(f"\n双力偶成分: {result['double_couple_percentage']:.1f}%")

        print("\n" + "="*60)

    def save_results(self, filename: str, format: str = 'json'):
        """保存结果到文件"""
        if not self.results:
            print("错误: 没有结果可保存")
            return

        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
        elif format == 'txt':
            with open(filename, 'w') as f:
                for result in self.results:
                    f.write(self._format_text_output(result))
                    f.write("\n" + "-"*40 + "\n")
        else:
            raise ValueError(f"不支持的格式: {format}")

        print(f"结果已保存到: {filename}")

    def _format_text_output(self, result: Dict) -> str:
        """格式化文本输出"""
        inp = result['input']
        mt = result['moment_tensor']

        text = f"Strike={inp['strike']:.1f} Dip={inp['dip']:.1f} Rake={inp['rake']:.1f}\n"
        text += f"Mrr={mt['Mrr']:.4e} Mtt={mt['Mtt']:.4e} Mpp={mt['Mpp']:.4e}\n"
        text += f"Mrt={mt['Mrt']:.4e} Mrp={mt['Mrp']:.4e} Mtp={mt['Mtp']:.4e}\n"

        return text

    def plot_beachball(self, result: Dict):
        """绘制震源球（如果matplotlib可用）"""
        if not PLOT_AVAILABLE:
            print("警告: matplotlib未安装，无法绘图")
            return

        try:
            from obspy.imaging.beachball import beachball

            mt = result['moment_tensor']
            mt_list = [mt['Mrr'], mt['Mtt'], mt['Mpp'], 
                      mt['Mrt'], mt['Mrp'], mt['Mtp']]

            fig = plt.figure(figsize=(6, 6))
            beachball(mt_list, size=200, linewidth=2, facecolor='red')

            inp = result['input']
            plt.title(f"Strike={inp['strike']:.0f}° "
                     f"Dip={inp['dip']:.0f}° "
                     f"Rake={inp['rake']:.0f}°")
            plt.show()

        except ImportError:
            print("警告: obspy未安装，无法绘制震源球")

def interactive_mode():
    """交互式模式"""
    converter = FocalMechanismToMomentTensor()

    print("\n" + "="*60)
    print("震源机制解到矩张量转换程序 - 交互式模式")
    print("="*60)
    print("输入 'q' 退出，'h' 显示帮助")

    while True:
        print("\n请输入震源机制解参数:")

        try:
            # 走向
            strike_input = input("走向 (0-360°): ").strip()
            if strike_input.lower() == 'q':
                break
            elif strike_input.lower() == 'h':
                print_help()
                continue

            strike = float(strike_input)

            # 倾角
            dip = float(input("倾角 (0-90°): "))

            # 滑动角
            rake = float(input("滑动角 (-180-180°): "))

            # 震级（可选）
            mag_input = input("震级 Mw (可选，按回车跳过): ").strip()
            magnitude = float(mag_input) if mag_input else None

            # 转换
            result = converter.strike_dip_rake_to_mt(strike, dip, rake, magnitude)

            # 显示结果
            converter.print_results(result)

            # 是否保存
            save = input("\n是否保存结果? (y/n): ").strip().lower()
            if save == 'y':
                filename = input("输入文件名 (默认: result.json): ").strip()
                filename = filename or "result.json"
                converter.save_results(filename)

            # 是否绘图
            if PLOT_AVAILABLE:
                plot = input("是否绘制震源球? (y/n): ").strip().lower()
                if plot == 'y':
                    converter.plot_beachball(result)

        except ValueError as e:
            print(f"错误: {e}")
        except KeyboardInterrupt:
            print("\n程序被中断")
            break

    print("\n感谢使用！")

def batch_mode(input_file: str, output_file: str):
    """批处理模式"""
    converter = FocalMechanismToMomentTensor()

    print(f"读取输入文件: {input_file}")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            parts = line.split()
            if len(parts) < 3:
                print(f"警告: 第{i}行格式错误，跳过")
                continue

            strike = float(parts[0])
            dip = float(parts[1])
            rake = float(parts[2])
            magnitude = float(parts[3]) if len(parts) > 3 else None

            result = converter.strike_dip_rake_to_mt(strike, dip, rake, magnitude)
            print(f"处理第{i}行: Strike={strike}, Dip={dip}, Rake={rake}")

        except Exception as e:
            print(f"错误: 第{i}行处理失败 - {e}")

    # 保存结果
    format = 'json' if output_file.endswith('.json') else 'txt'
    converter.save_results(output_file, format=format)

def print_help():
    """打印帮助信息"""
    help_text = """
震源机制解参数说明:
--------------------
走向 (Strike): 断层走向，从北向顺时针测量，范围0-360度
倾角 (Dip): 断层面倾角，范围0-90度
滑动角 (Rake): 滑动方向与走向的夹角，范围-180到180度
  -90°: 纯正断层
   0°: 纯左旋走滑
  90°: 纯逆断层
  180°/-180°: 纯右旋走滑
坐标系统:
---------
USE: Up-South-East
r: 向上（径向）
t: 向南（切向）  
p: 向东（横向）
矩张量分量:
-----------
Mrr: 垂直-垂直
Mtt: 南-南
Mpp: 东-东
Mrt: 垂直-南
Mrp: 垂直-东
Mtp: 南-东
"""
    print(help_text)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='震源机制解到矩张量转换程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  交互式模式:
    python focal_to_mt.py

  命令行模式:
    python focal_to_mt.py --strike 225 --dip 45 --rake -90
    python focal_to_mt.py --strike 225 --dip 45 --rake -90 --magnitude 5.5

  批处理模式:
    python focal_to_mt.py --input mechanisms.txt --output results.json

输入文件格式 (mechanisms.txt):
  # strike dip rake [magnitude]
  225 45 -90 5.5
  180 60 0 4.2
  90 30 90
        """
    )

    parser.add_argument('--strike', type=float, help='走向 (0-360度)')
    parser.add_argument('--dip', type=float, help='倾角 (0-90度)')
    parser.add_argument('--rake', type=float, help='滑动角 (-180-180度)')
    parser.add_argument('--magnitude', type=float, help='矩震级 Mw (可选)')
    parser.add_argument('--input', type=str, help='输入文件 (批处理模式)')
    parser.add_argument('--output', type=str, help='输出文件')
    parser.add_argument('--plot', action='store_true', help='绘制震源球')
    parser.add_argument('--detailed', action='store_true', 
                       default=True, help='显示详细结果')

    args = parser.parse_args()

    # 批处理模式
    if args.input:
        if not args.output:
            args.output = 'results.json'
        batch_mode(args.input, args.output)

    # 命令行模式
    elif args.strike is not None and args.dip is not None and args.rake is not None:
        converter = FocalMechanismToMomentTensor()
        result = converter.strike_dip_rake_to_mt(
            args.strike, args.dip, args.rake, args.magnitude
        )
        converter.print_results(result, detailed=args.detailed)

        if args.output:
            converter.save_results(args.output)

        if args.plot:
            converter.plot_beachball(result)

    # 交互式模式
    else:
        interactive_mode()

if __name__ == "__main__":
    main()