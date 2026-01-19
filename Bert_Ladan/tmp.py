import pickle
import pandas as pd

def preview_pkl_file(file_path, max_items=10):
    """
    读取并预览pkl文件内容
    
    Args:
        file_path: pkl文件路径
        max_items: 最多显示的项目数量
    """
    try:
        print(f"正在读取文件: {file_path}")
        
        # 读取pkl文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n数据类型: {type(data)}")
        
        # 根据不同数据类型进行预览
        if isinstance(data, pd.DataFrame):
            print(f"数据形状: {data.shape}")
            print("\n前{max_items}行数据:")
            print(data.head(max_items))
            
        elif isinstance(data, dict):
            print(f"字典键数量: {len(data)}")
            print(f"字典键: {list(data.keys())}")
            
            # 显示前几个键值对
            print(f"\n前{min(max_items, len(data))}个键值对:")
            for i, (key, value) in enumerate(data.items()):
                if i >= max_items:
                    break
                print(f"  {key}: {type(value)} - {str(value)[:100]}...")
                
        elif isinstance(data, list):
            print(f"列表长度: {len(data)}")
            print(f"\n前{min(max_items, len(data))}个元素:")
            for i, item in enumerate(data[:max_items]):
                print(f"  [{i}]: {type(item)} - {str(item)[:100]}...")
                
        elif isinstance(data, str):
            print(f"字符串长度: {len(data)}")
            print(f"前{max_items*50}个字符:")
            print(data[:max_items*50])
            
        else:
            print(f"数据内容预览:")
            print(str(data)[:max_items*100])
            
        # 如果是大型数据结构，显示内存占用
        if hasattr(data, '__sizeof__'):
            size_mb = data.__sizeof__() / (1024*1024)
            print(f"\n预估内存占用: {size_mb:.2f} MB")
            
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
    except pickle.UnpicklingError:
        print("错误: 文件格式不正确或已损坏")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    file_path = "train_processed_thulac.pkl"
    preview_pkl_file(file_path, max_items=5)