import numpy as np
import paddle.inference as paddle_infer


def create_predictor():
    config = paddle_infer.Config("./resnet50/inference.pdmodel", "./resnet50/inference.pdiparams")
    config.enable_memory_optim()
    config.enable_use_gpu(1000, 0)

    # 打开TensorRT
    config.enable_tensorrt_engine(workspace_size=1 << 30,
                                  max_batch_size=1,
                                  min_subgraph_size=3,
                                  precision_mode=paddle_infer.PrecisionType.Float32,
                                  use_static=False, use_calib_mode=False)

    predictor = paddle_infer.create_predictor(config)
    return predictor


def run(predictor, img):
    # 准备输入
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())
    # 推理
    predictor.run()
    results = []
    # 获取输出
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results


if __name__ == '__main__':
    pred = create_predictor()
    img = np.ones((1, 3, 224, 224)).astype(np.float32)
    result = run(pred, [img])
    print("class index: ", np.argmax(result[0][0]))