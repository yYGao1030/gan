import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def clean_vibration_blur(image, intensity=3.0, frequency=5):
    """
    """
    h, w = image.shape[:2]
    
    # 生成平滑的震动轨迹 (使用余弦函数保证平滑过渡)
    t = np.linspace(0, 2*np.pi, frequency)
    x_shifts = intensity * np.cos(t) * random.uniform(0.8, 1.2)
    y_shifts = intensity * np.sin(t + np.pi/4) * random.uniform(0.8, 1.2)
    
    # 使用浮点运算避免累积误差
    accumulated = np.zeros_like(image, dtype=np.float32)
    
    for dx, dy in zip(x_shifts, y_shifts):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image.astype(np.float32), M, (w, h), 
                               borderMode=cv2.BORDER_REFLECT)
        accumulated += shifted
    
    # 平均帧并转换回uint8
    blurred = np.clip(accumulated / frequency, 0, 255).astype(np.uint8)
    
    # 添加灰度传感器噪声 (避免彩色噪点)
    if random.random() < 0.4:
        # 只在亮度通道添加噪声
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        noise = np.random.normal(0, intensity/4, hsv[:,:,2].shape)
        hsv[:,:,2] = np.clip(hsv[:,:,2] + noise, 0, 255)
        blurred = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return blurred

def create_clean_vibration_dataset(input_dir, output_dir, num_pairs=1000):
    """
    """
    # 创建输出目录
    sharp_dir = os.path.join(output_dir, 'trainB')
    blur_dir = os.path.join(output_dir, 'trainA')
    os.makedirs(sharp_dir, exist_ok=True)
    os.makedirs(blur_dir, exist_ok=True)
    
    # 获取输入图像并过滤无效文件
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = [f for f in image_files 
                  if not os.path.isdir(os.path.join(input_dir, f))]
    
    if not image_files:
        raise ValueError("输入目录中没有有效的图像文件")
    
    # 生成图像对
    for i in tqdm(range(num_pairs), desc="生成无噪点震动模糊数据集"):
        img_file = random.choice(image_files)
        img_path = os.path.join(input_dir, img_file)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 预处理 (保持宽高比)
            h, w = img.shape[:2]
            crop_size = min(h, w, 512)
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            img_cropped = img[y:y+crop_size, x:x+crop_size]
            img_resized = cv2.resize(img_cropped, (256, 256))
            
            # 生成无彩色噪点的震动模糊
            intensity = random.uniform(1.5, 3.5)
            frequency = random.randint(4, 7)
            blurred = clean_vibration_blur(img_resized, intensity, frequency)
            
            # 保存图像对 (PNG格式避免JPEG压缩伪影)
            pair_id = f"clean_vib_{i:04d}"
            cv2.imwrite(os.path.join(sharp_dir, f"{pair_id}.png"), img_resized)
            cv2.imwrite(os.path.join(blur_dir, f"{pair_id}.png"), blurred)
            
        except Exception as e:
            print(f"处理 {img_file} 时出错: {str(e)}")
            continue
    
    print(f"数据集生成完成! 清晰图像: {sharp_dir}")
    print(f"          震动模糊图像: {blur_dir}")

def visualize_clean_samples(dataset_dir, num_samples=5):
    """可视化无噪点样本"""
    sharp_dir = os.path.join(dataset_dir, 'trainB')
    blur_dir = os.path.join(dataset_dir, 'trainA')
    
    sharp_files = sorted([f for f in os.listdir(sharp_dir) if f.endswith('.png')])[:num_samples]
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, file in enumerate(sharp_files):
        sharp_path = os.path.join(sharp_dir, file)
        blur_path = os.path.join(blur_dir, file)
        
        sharp_img = cv2.imread(sharp_path)
        blur_img = cv2.imread(blur_path)
        
        # 转换为RGB
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        
        # 计算颜色通道标准差 (检测彩色噪点)
        sharp_std = [sharp_img[...,c].std() for c in range(3)]
        blur_std = [blur_img[...,c].std() for c in range(3)]
        
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(sharp_img)
        plt.title(f"清晰图像\nRGB标准差: {sharp_std}")
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(blur_img)
        plt.title(f"震动模糊\nRGB标准差: {blur_std}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, 'clean_samples.png'), bbox_inches='tight', dpi=300)
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 配置路径
    CLEAR_IMAGES_DIR = "/home/gyy/gan_ws/deblur_gan/images/train/B/"  # 替换为你的无人机清晰图像目录
    OUTPUT_DIR = "uav_clean_vibration_dataset"     # 输出目录
    NUM_PAIRS = 2000                              # 图像对数量
    
    # 生成无彩色噪点的震动模糊数据集
    create_clean_vibration_dataset(CLEAR_IMAGES_DIR, OUTPUT_DIR, NUM_PAIRS)
    
    # 可视化结果验证
    visualize_clean_samples(OUTPUT_DIR)
