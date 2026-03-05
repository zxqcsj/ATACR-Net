# Enhanced HDGCN Configuration Examples
# 增强版HDGCN配置示例

def get_enhanced_hdgcn_config():
    """推荐的增强版HDGCN配置"""
    return {
        'model': 'model.HDGCN.Model',
        'model_args': {
            'num_class': 60,
            'num_point': 25,
            'num_person': 2,
            'graph': 'graph.ntu_rgb_d_hierarchy.Graph',
            'graph_args': {
                'labeling_mode': 'spatial',
                'CoM': 21
            },
            'base_channel': 64,
            'adaptive': True,
            'drop_out': 0.0,
            
            # 新增架构参数
            'architecture': 'enhanced_hdgcn',
            'temporal_modeling_config': {
                'early_layers': 'mctm',      # 前期使用MCTM时序建模
                'middle_layers': 'standard',  # 中期使用标准时序卷积
                'late_layers': 'standard',   # 后期使用标准时序卷积
                'transition_layer': 4        # 从第4层开始过渡
            },
            'attention_config': {
                'start_layer': 4,           # 从第4层开始使用注意力
                'use_se_attention': True,   # 使用SE注意力
                'use_aha_attention': True,  # 使用AHA注意力
                'attention_reduction': 16   # SE注意力降维比例
            },
            'fusion_config': {
                'mode': 'adaptive',         # 自适应融合
                'use_learnable_weights': True,  # 使用可学习权重
                'spatial_weight': 1.0,      # 空间特征权重
                'temporal_weight': 0.1      # 时序特征权重
            }
        }
    }

def get_original_hdgcn_config():
    """原始HDGCN配置"""
    return {
        'model': 'model.HDGCN.Model',
        'model_args': {
            'num_class': 60,
            'num_point': 25,
            'num_person': 2,
            'graph': 'graph.ntu_rgb_d_hierarchy.Graph',
            'graph_args': {
                'labeling_mode': 'spatial',
                'CoM': 21
            },
            'base_channel': 64,
            'adaptive': True,
            'drop_out': 0.0,
            
            # 使用原始HDGCN架构
            'architecture': 'original_hdgcn',
        }
    }

def get_hybrid_config():
    """混合架构配置 - 兼容当前实现"""
    return {
        'model': 'model.HDGCN.Model',
        'model_args': {
            'num_class': 60,
            'num_point': 25,
            'num_person': 2,
            'graph': 'graph.ntu_rgb_d_hierarchy.Graph',
            'graph_args': {
                'labeling_mode': 'spatial',
                'CoM': 21
            },
            'base_channel': 64,
            'adaptive': True,
            'drop_out': 0.0,
            
            # 混合架构
            'architecture': 'hybrid',
            'temporal_modeling_config': {
                'early_layers': 'motion',    # 前期使用Motion GCN
                'middle_layers': 'standard', # 中期过渡
                'late_layers': 'standard',   # 后期使用标准HDGCN
                'transition_layer': 5        # 从第5层开始过渡
            }
        }
    }

def get_backward_compatible_config():
    """向后兼容配置 - 使用旧参数"""
    return {
        'model': 'model.HDGCN.Model',
        'model_args': {
            'num_class': 60,
            'num_point': 25,
            'num_person': 2,
            'graph': 'graph.ntu_rgb_d_hierarchy.Graph',
            'graph_args': {
                'labeling_mode': 'spatial',
                'CoM': 21
            },
            'base_channel': 64,
            'adaptive': True,
            'drop_out': 0.0,
            
            # 旧参数 - 会自动转换为新架构
            'modality': 2,              # Motion模态
            'use_mctm_tcn': True,       # 使用MCTM时序卷积
        }
    }

# 使用示例
if __name__ == '__main__':
    # 获取推荐配置
    config = get_enhanced_hdgcn_config()
    print("Enhanced HDGCN Config:")
    print(config)
    
    # 获取原始HDGCN配置
    original_config = get_original_hdgcn_config()
    print("\nOriginal HDGCN Config:")
    print(original_config)
    
    # 获取混合配置
    hybrid_config = get_hybrid_config()
    print("\nHybrid Config:")
    print(hybrid_config)
