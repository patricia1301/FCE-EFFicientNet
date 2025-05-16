import torch
import torch.nn as nn


class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        print(x_h.shape, x_w.shape)
        # 在两个维度上应用注意力
        return x * x_h * x_w


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNGroup(nn.Module):
    def __init__(self, channels, kernel_size):
        super(ConvBNGroup, self).__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )
        self.gn = nn.GroupNorm(16, channels)

    def forward(self, x):
        return self.gn(self.conv(x))


class MultiScaleELA(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 7, 11]):
        super(MultiScaleELA, self).__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes

        # Height dimension processing branch
        self.h_convs = nn.ModuleList([
            ConvBNGroup(channels, k) for k in kernel_sizes
        ])

        # Width dimension processing branch
        self.w_convs = nn.ModuleList([
            ConvBNGroup(channels, k) for k in kernel_sizes
        ])

        # 1x1 convs after concatenation
        self.h_conv1x1 = nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1)
        self.w_conv1x1 = nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1)

        # SE Block
        self.se_fc1 = nn.Linear(channels * 2, channels // 4)
        self.se_relu = nn.ReLU(inplace=True)
        self.se_fc2 = nn.Linear(channels // 4, channels)
        self.se_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Height dimension processing
        x_h = torch.mean(x, dim=3).view(b, c, h)  # Average pooling along width
        h_outs = []
        for conv in self.h_convs:
            h_outs.append(conv(x_h))
        h_concat = torch.cat(h_outs, dim=1)
        h_out = self.h_conv1x1(h_concat).view(b, c, h, 1)

        # Width dimension processing
        x_w = torch.mean(x, dim=2).view(b, c, w)  # Average pooling along height
        w_outs = []
        for conv in self.w_convs:
            w_outs.append(conv(x_w))
        w_concat = torch.cat(w_outs, dim=1)
        w_out = self.w_conv1x1(w_concat).view(b, c, 1, w)

        # Combine both branches
        combined = x * h_out * w_out

        # Global Average Pooling for SE Block
        se_input = torch.cat([
            F.adaptive_avg_pool2d(h_out, 1).view(b, c),
            F.adaptive_avg_pool2d(w_out, 1).view(b, c)
        ], dim=1)

        # SE Block
        se_out = self.se_fc1(se_input)
        se_out = self.se_relu(se_out)
        se_out = self.se_fc2(se_out)
        se_out = self.se_sigmoid(se_out).view(b, c, 1, 1)

        # Apply SE attention
        output = combined * se_out

        # Skip connection
        output = output + x

        return output


# Simplified version closer to the provided ELA code
class ELA(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(ELA, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size,
                              padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Process height dimension
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # Process width dimension
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # Apply attention in both dimensions
        return x * x_h * x_w


# Enhanced Multi-Scale ELA that more directly matches the diagram
class MultiScaleELAEnhanced(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 7, 11]):
        super(MultiScaleELAEnhanced, self).__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes

        # Height dimension processing branch
        self.h_convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)
            gn = nn.GroupNorm(16, channels)
            self.h_convs.append(nn.Sequential(conv, gn))

        # Width dimension processing branch
        self.w_convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)
            gn = nn.GroupNorm(16, channels)
            self.w_convs.append(nn.Sequential(conv, gn))

        # 1x1 convs after concatenation
        self.h_conv1x1 = nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1)
        self.w_conv1x1 = nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1)

        # SE Block
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // 16, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Height dimension processing
        x_h_pool = torch.mean(x, dim=3).view(b, c, h)  # (B, C, H)
        h_outs = []
        for conv in self.h_convs:
            h_outs.append(conv(x_h_pool))
        h_concat = torch.cat(h_outs, dim=1)  # (B, 3C, H)
        h_out = self.h_conv1x1(h_concat).view(b, c, h, 1)  # (B, C, H, 1)

        # Width dimension processing
        x_w_pool = torch.mean(x, dim=2).view(b, c, w)  # (B, C, W)
        w_outs = []
        for conv in self.w_convs:
            w_outs.append(conv(x_w_pool))
        w_concat = torch.cat(w_outs, dim=1)  # (B, 3C, W)
        w_out = self.w_conv1x1(w_concat).view(b, c, 1, w)  # (B, C, 1, W)

        # Multiply input with both attention maps
        combined = x * h_out * w_out

        # SE Block
        se_out = self.global_pool(combined).view(b, c)
        se_out = self.fc1(se_out)
        se_out = self.relu(se_out)
        se_out = self.fc2(se_out)
        se_out = self.sigmoid(se_out).view(b, c, 1, 1)

        # Final output with SE attention and skip connection
        output = combined * se_out + x

        return output


# Example usage
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size, channels, height, width = 2, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)

    # Initialize the ELA module
    ela = ELA(channels)
    output = ela(x)
    print(f"ELA output shape: {output.shape}")

    # Initialize the Multi-Scale ELA module
    multi_ela = MultiScaleELA(channels)
    output = multi_ela(x)
    print(f"Multi-Scale ELA output shape: {output.shape}")

    # Initialize the Enhanced Multi-Scale ELA module
    multi_ela_enhanced = MultiScaleELAEnhanced(channels)
    output = multi_ela_enhanced(x)
    print(f"Enhanced Multi-Scale ELA output shape: {output.shape}")

    import torch
    import torch.nn as nn
    import time
    import torchvision.models as models


    def compute_model_metrics(model, input_tensor, num_iterations=100):
        """
        计算给定模型的参数数量、每秒浮点运算量 (GPLOPS) 和每秒帧数 (FPS)

        :param model: PyTorch 模型
        :param input_tensor: 输入张量
        :param num_iterations: 测量的推理次数，默认为 100
        :return: 返回模型的 params, GPLOPS 和 FPS
        """

        # 确保模型在评估模式
        model.eval()
        model = model.cuda()
        input_tensor = input_tensor.cuda()

        # 计算模型参数数量
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        num_params = count_parameters(model)

        # 计算推理时间，计算FPS
        total_time = 0.0
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)
            end_time = time.time()
            total_time += (end_time - start_time)

        average_inference_time = total_time / num_iterations
        fps = num_iterations / total_time

        # 估算GPLOPS（假设ResNet18大约1.8亿次浮点运算）
        # 实际模型的FLOPS可以通过更多的方式估算，这里简化处理
        flops = 1.8 * 10 ** 8  # 示例：ResNet18大致FLOPS数量
        gflops = flops / (average_inference_time * 10 ** 9)

        return num_params, gflops, fps


    # 示例：加载模型并传入输入张量


    for model in [ela, multi_ela, multi_ela_enhanced]:
        input_tensor = torch.randn(2, 64, 32, 32)  # 示例输入

        # 计算模型的 params, GPLOPS, 和 FPS
        num_params, gflops, fps = compute_model_metrics(model, input_tensor)

        # 打印结果
        print(f"模型参数数量: {num_params}")
        print(f"模型的GPLOPS: {gflops:.2f}")
        print(f"模型的FPS: {fps:.2f} 帧每秒")



