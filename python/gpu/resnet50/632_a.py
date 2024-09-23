import paddle

# 创建 PaddlePaddle 执行环境
place = paddle.CPUPlace()  # CPU 上执行
exe = paddle.static.Executor(place)

paddle.enable_static()

# 模型路径
model_dir = "./resnet50/"

# 加载模型和参数
inference_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
    model_dir + "inference",
    exe)

# 打印模型输入输出信息
print("模型输入:", feed_target_names)
print("模型输出:", fetch_targets)