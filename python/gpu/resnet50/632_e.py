import numpy as np
import paddle.inference as paddle_infer

def create_predictor():
    config = paddle_infer.Config("./resnet50/inference.pdmodel", "./resnet50/inference.pdiparams")
    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0)
    predictor = paddle_infer.create_predictor(config)
    return predictor

def preprocess_image(img):
    # 假设 img 的值在 [0, 255] 范围内，归一化到 [0, 1]
    img = img / 255.0
    # 模型可能还需要均值减法或者其他预处理操作
    # 例如：
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # img = (img - mean) / std
    return img

def run(predictor, img):
    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 打印输入数据的形状,即转换前旧的 shape 信息
    print("Original image shape:", img.shape)

    # 预处理图像
    img = preprocess_image(img)

    # Reshape 输入 tensor 并从 CPU 复制数据
    input_handle.reshape(img.shape)
    input_handle.copy_from_cpu(img)

    # 验证 reshape 之后的输入张量形状，即转换后新的 shape
    print("Reshaped input tensor shape:", input_handle.shape())
    predictor.run()
    results = []
    output_names = predictor.get_output_names()
    for name in output_names:
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


if __name__ == '__main__':
    pred = create_predictor()
    img = np.ones((1, 3, 384, 224)).astype(np.float32)

    result = run(pred, img)
    print("Output shape:", result[0].shape)
    print("Predicted class index:", np.argmax(result[0][0]))