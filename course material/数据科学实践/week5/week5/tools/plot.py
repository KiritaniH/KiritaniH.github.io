import matplotlib.pyplot as plt
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns





# # Define the classes (assuming they are defined as in your previous code)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_images(dataloader, classes=None, num_images=None):
    """
    Display a grid of random images and their labels from a DataLoader.
    
    Args:
        dataloader: A PyTorch DataLoader object (e.g., trainloader or testloader)
        num_images (int, optional): Number of images to display. If None, use batch_size from dataloader.
                                   Defaults to None, which uses the full batch.
    
    Returns:
        None: Displays the images and prints their labels.
    """
    # Get an iterator for the DataLoader
    dataiter = iter(dataloader)
    
    # Get the next batch of images and labels
    images, labels = next(dataiter)
    
    # Determine the number of images to display (default to batch size if num_images is None)
    if num_images is None:
        num_images = len(images)  # Use the full batch size
    
    # Ensure num_images doesn't exceed the batch size
    num_images = min(num_images, len(images))
    
    # Unnormalize the images (assuming normalization with mean=0.5, std=0.5 as in your transform)
    images = images / 2 + 0.5
    
    # Convert images to numpy and transpose for plotting (C, H, W) -> (H, W, C)
    npimages = images.numpy()
    npimages = np.transpose(npimages[:num_images], (0, 2, 3, 1))
    
    # Create a grid of images using torchvision.utils.make_grid
    grid_img = torchvision.utils.make_grid(images[:num_images])
    npgrid_img = grid_img.numpy()
    npgrid_img = np.transpose(npgrid_img, (1, 2, 0))
    
    # Display the grid of images
    plt.figure(figsize=(10, 5))  # Optional: Adjust figure size for better visualization
    plt.imshow(npgrid_img)
    plt.axis('off')  # Hide axes
    # plt.title('Random Images from DataLoader')
    plt.show()
    
    # Print the labels for the displayed images
    if classes is not None:
        label_texts = ' '.join(f'{classes[labels[j]]:5s}' for j in range(num_images))
        print(f'Labels: {label_texts}')





def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    """
    绘制训练损失、训练准确率、测试损失、测试准确率和学习率曲线
    """
    num_epochs = len(train_losses)
    # 创建 1 行 4 列的子图
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))  # 1 行 4 列

    # 绘制训练损失
    ax[0].plot(range(1, num_epochs+1), train_losses, label='Train Loss', color='blue')
    ax[0].set_title('Train Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)

    # 绘制训练准确率
    ax[1].plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy', color='green')
    ax[1].set_title('Train Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid(True)

    # 绘制测试损失
    ax[2].plot(range(1, num_epochs+1), test_losses, label='Test Loss', color='red')
    ax[2].set_title('Test Loss')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Loss')
    ax[2].grid(True)

    # 绘制测试准确率
    ax[3].plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy', color='orange')
    ax[3].set_title('Test Accuracy')
    ax[3].set_xlabel('Epochs')
    ax[3].set_ylabel('Accuracy')
    ax[3].grid(True)

    plt.tight_layout()
    plt.show()


# 获取训练和测试数据的类别分布
def get_class_distribution(dataset):
    """
    获取数据集的类别分布。
    
    Args:
        dataset: PyTorch Dataset 对象（如 CIFAR-10 数据集）
    
    Returns:
        np.ndarray: 每个类别的样本数量
    """
    class_counts = np.zeros(10, dtype=int)
    for _, label in dataset:
        class_counts[label] += 1
    return class_counts


def plot_cifar_distribution(trainset, testset, classes, filename='cifar10_distribution.png'):
    """
    绘制 CIFAR-10 训练和测试数据的类别分布柱状图，使用 matplotlib。
    
    Args:
        trainset: 训练数据集
        testset: 测试数据集
        classes: 类别名称列表
        filename: 保存图像的文件名
    """
    # 获取分布
    train_distribution = get_class_distribution(trainset)
    test_distribution = get_class_distribution(testset)
    
    # 设置图形大小
    plt.figure(figsize=(12, 5))
    
    # 定义颜色列表（使用 matplotlib 的默认颜色循环）
    colors = plt.cm.tab10.colors  # 10 种颜色
    
    # 绘制训练数据分布
    plt.subplot(1, 2, 1)
    plt.bar(range(10), train_distribution, color=colors, alpha=0.7)
    plt.title('Distribution of Training Data')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(range(10), classes, rotation=45)
    
    # 绘制测试数据分布
    plt.subplot(1, 2, 2)
    plt.bar(range(10), test_distribution, color=colors, alpha=0.7)
    plt.title('Distribution of Testing Data')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(range(10), classes, rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # # 保存图像
    # if filename:
    #     plt.savefig(filename)
    
    # 显示图形
    plt.show()