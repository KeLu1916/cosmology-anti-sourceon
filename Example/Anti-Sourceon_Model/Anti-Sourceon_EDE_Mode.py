# antiosourceon_final_summary.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 显示中文字体
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False

class AntiSourceonFinalSummary:
    def __init__(self, best_params):
        self.best_params = best_params
        self.planck_params = {
            'H0': 67.66, 'ombh2': 0.02242, 'omch2': 0.11933,
            'tau': 0.0561, 'As': 2.105e-9, 'ns': 0.9665
        }
        
        # 反源子物理参数
        self.antiosourceon_physics = {
            'z_dec': 5161,      # 退耦红移
            'f_ede': 0.1,       # EDE峰值占比
            'DeltaE': 2.07,     # 能垒 (eV)
            'g_int': 6.95e-5,   # 耦合强度
            'm_anti': 100.0     # 反源子质量 (eV)
        }

    def calculate_cosmological_quantities(self):
        """计算关键宇宙学量"""
        # 当前模型
        omega_m_our = (self.best_params['ombh2'] + self.best_params['omch2']) / (self.best_params['H0']/100)**2
        omega_lambda_our = 1 - omega_m_our
        
        # Planck模型
        omega_m_planck = (self.planck_params['ombh2'] + self.planck_params['omch2']) / (self.planck_params['H0']/100)**2
        omega_lambda_planck = 1 - omega_m_planck
        
        # 变化百分比
        h0_change = ((self.best_params['H0'] - self.planck_params['H0']) / self.planck_params['H0']) * 100
        omega_m_change = ((omega_m_our - omega_m_planck) / omega_m_planck) * 100
        
        return {
            '我们的模型': {
                'H0': self.best_params['H0'],
                'Ω_m': omega_m_our,
                'Ω_Λ': omega_lambda_our,
                'Ω_b': self.best_params['ombh2'] / (self.best_params['H0']/100)**2,
                'Ω_c': self.best_params['omch2'] / (self.best_params['H0']/100)**2
            },
            'Planck ΛCDM': {
                'H0': self.planck_params['H0'],
                'Ω_m': omega_m_planck,
                'Ω_Λ': omega_lambda_planck,
                'Ω_b': self.planck_params['ombh2'] / (self.planck_params['H0']/100)**2,
                'Ω_c': self.planck_params['omch2'] / (self.planck_params['H0']/100)**2
            },
            '变化': {
                'ΔH₀': h0_change,
                'ΔΩ_m': omega_m_change
            }
        }

    def plot_comprehensive_summary(self, cosmology_quantities):
        """绘制综合总结图"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 哈勃张力解决进度
        plt.subplot(3, 3, 1)
        measurements = {
            'Planck ΛCDM': 67.66,
            '反源子EDE': self.best_params['H0'],
            'SH0ES (目标)': 73.04,
            '中间值': 70.35
        }
        errors = {
            'Planck ΛCDM': 0.42,
            '反源子EDE': 1.0,  # 我们的估计误差

        }
        
        colors = ['blue', 'red', 'green', 'orange']
        positions = range(len(measurements))
        
        for i, (label, value) in enumerate(measurements.items()):
            plt.barh(i, value, xerr=errors.get(label, 0), 
                    color=colors[i], alpha=0.7, label=label)
        
        plt.xlabel('H₀ [km/s/Mpc]')
        plt.yticks(positions, measurements.keys())
        plt.title('哈勃张力解决进度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 参数变化对比
        plt.subplot(3, 3, 2)
        parameters = ['H₀', 'Ω_m', 'Ω_Λ', 'Ω_b', 'Ω_c']
        our_values = [
            cosmology_quantities['我们的模型']['H0'],
            cosmology_quantities['我们的模型']['Ω_m'],
            cosmology_quantities['我们的模型']['Ω_Λ'],
            cosmology_quantities['我们的模型']['Ω_b'],
            cosmology_quantities['我们的模型']['Ω_c']
        ]
        planck_values = [
            cosmology_quantities['Planck ΛCDM']['H0'],
            cosmology_quantities['Planck ΛCDM']['Ω_m'],
            cosmology_quantities['Planck ΛCDM']['Ω_Λ'],
            cosmology_quantities['Planck ΛCDM']['Ω_b'],
            cosmology_quantities['Planck ΛCDM']['Ω_c']
        ]
        
        x_pos = np.arange(len(parameters))
        width = 0.35
        
        plt.bar(x_pos - width/2, our_values, width, label='反源子EDE', alpha=0.7)
        plt.bar(x_pos + width/2, planck_values, width, label='Planck ΛCDM', alpha=0.7)
        plt.xticks(x_pos, parameters)
        plt.ylabel('参数值')
        plt.title('宇宙学参数对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 反源子物理机制
        plt.subplot(3, 3, 3)
        plt.axis('off')
        
        physics_text = (
            "反源子模型物理机制:\n\n"
            "基本参数:\n"
            f"• 退耦红移: z = {self.antiosourceon_physics['z_dec']}\n"
            f"• EDE峰值: f_EDE = {self.antiosourceon_physics['f_ede']:.2f}\n"
            f"• 相互作用能垒: ΔE = {self.antiosourceon_physics['DeltaE']} eV\n"
            f"• 耦合强度: g = {self.antiosourceon_physics['g_int']:.2e}\n"
            f"• 反源子质量: m = {self.antiosourceon_physics['m_anti']} eV\n\n"
            "物理过程:\n"
            "1. 早期宇宙反源子活跃\n"
            "2. 提供早期暗能量\n"
            "3. z≈5000时退耦\n"
            "4. 成为冷暗物质\n"
            "5. 提高H₀估计值"
        )
        
        plt.text(0.05, 0.95, physics_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        
       
        
        # 6. 参数敏感性分析
        plt.subplot(3, 3, 7)
        parameters = ['H₀', 'Ω_bh²', 'Ω_ch²', 'τ', 'A_s', 'n_s']
        our_params = [
            self.best_params['H0'],
            self.best_params['ombh2'],
            self.best_params['omch2'],
            self.best_params['tau'], 
            self.best_params['As'] * 1e9,
            self.best_params['ns']
        ]
        planck_params = [
            self.planck_params['H0'],
            self.planck_params['ombh2'],
            self.planck_params['omch2'],
            self.planck_params['tau'],
            self.planck_params['As'] * 1e9, 
            self.planck_params['ns']
        ]
        
        changes = [(o-p)/p*100 for o, p in zip(our_params, planck_params)]
        
        colors = ['red' if abs(c) > 1 else 'blue' for c in changes]
        plt.bar(parameters, changes, color=colors, alpha=0.7)
        plt.axhline(0, color='black', linestyle='-')
        plt.ylabel('相对变化 (%)')
        plt.title('参数相对变化')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. 宇宙学时间线
        plt.subplot(3, 3, 8)
        events = {
            '暴胀结束': 1e-32,
            '原初核合成': 1e-3, 
            '反源子退耦': 5e4,
            '复合时期': 1.1e3,
            '再电离': 10,
            '今天': 0
        }
        
        plt.semilogy(list(events.values()), list(events.keys()), 'o-', linewidth=2)
        plt.axvline(self.antiosourceon_physics['z_dec'], color='red', linestyle='--', 
                   label='反源子退耦')
        plt.xlabel('红移 z')
        plt.title('宇宙学时间线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.show()

    def generate_final_report(self):
        """生成最终报告"""
        print("=" * 70)
        print("             反源子宇宙学模型 - 最终研究报告")
        print("=" * 70)
        
        # 计算宇宙学量
        cosmology = self.calculate_cosmological_quantities()
        
        print("\n📊 关键结果:")
        print(f"  哈勃常数: H₀ = {self.best_params['H0']:.2f} km/s/Mpc")
        print(f"  相对于Planck提升: {cosmology['变化']['ΔH₀']:+.1f}%")
        print(f"  物质密度: Ω_m = {cosmology['我们的模型']['Ω_m']:.4f}")
        print(f"  暗能量密度: Ω_Λ = {cosmology['我们的模型']['Ω_Λ']:.4f}")
        

    
        

        
        # 绘制综合图表
        self.plot_comprehensive_summary(cosmology)
        
        return cosmology

# 使用最佳参数生成最终报告
if __name__ == "__main__":
    # 从优化结果中获取最佳参数
    best_params = {
        'H0': 69.59, 'ombh2': 0.0223, 'omch2': 0.118,
        'tau': 0.058, 'As': 2.12e-9, 'ns': 0.968
    }
    
    summary = AntiSourceonFinalSummary(best_params)
    final_results = summary.generate_final_report()
    
    print("\n" + "=" * 70)
    print("研究完成! 反源子EDE模型构建成功!")
    print("=" * 70)