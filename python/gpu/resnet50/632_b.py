import paddle
import paddle.static as static

paddle.enable_static()

# 定义输入
x = static.data(name='x', shape=[None, 3, 224, 224], dtype='float32')

# 定义模型结构，例如使用一个简单的卷积层
conv2d = paddle.static.nn.conv2d(input=x, num_filters=64, filter_size=3)

# 定义损失函数或进一步的网络层（可选）
output = paddle.nn.functional.relu(conv2d)

# 获取执行器
place = paddle.CPUPlace()  # 或者 GPU place = paddle.CUDAPlace(0)
exe = static.Executor(place)

# 初始化参数
exe.run(static.default_startup_program())

# 打印计算图信息
print(static.default_main_program())