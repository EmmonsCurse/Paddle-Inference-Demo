
# Paddle Inference 支持动态图 & 静态图混合推理装饰器使用方法

## 1. 相关背景

长久以来，Paddle Inference 推理流程是如下代码所示的三个步骤:  

* 步骤1. 算法人员在**动态图组网**上完成组网代码和训练   
* 步骤2. 部署人员开发**动转静脚本**将模型转化成静态图模型，保存到磁盘上  
* 步骤3. 部署人员用python/C++ API开发**静态图推理脚本**  

```py
import paddle
#步骤一:模型组网代码  
class ExampleLayer(paddle.nn.Layer):
    def __init__(self, hidd):
        super().__init__()
        self.fn = paddle.nn.Linear(hidd, hidd, bias_attr=False)
    def forward(self, x):
        for i in range(10):
            x = paddle.nn.functional.softmax(x,-1)
        x = x.cast("float32")
        x = self.func(x)
        return x
    def func(self, x):
        x = x + x
        return self.fn(x)

#步骤二:动转静代码  
batch = 4096
hidd = 1024
dtype = "bfloat16"
x = paddle.rand([batch, hidd], dtype=dtype)

mylayer = ExampleLayer(hidd)

model = paddle.jit.to_static(
    mylayer,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None, None], dtype=dtype),
    ])

# save to static model
save_path = "./checkpoints/infer"
paddle.jit.save(model, save_path)
print(f"static model has been to {save_path}")

#步骤三:静态图推理代码  
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

model_dir = "checkpoints/"
model_file = model_dir + "/infer.pdmodel"
params_file = model_dir + "/infer.pdiparams"
config = Config(model_file, params_file)
config.enable_memory_optim()
gpu_precision = PrecisionType.Float32
config.enable_use_gpu(1000, 0, gpu_precision)
predictor = create_predictor(config)
result = predictor.run([x])
``` 

上文是三个步骤的代码，当遇到复杂模型时，步骤2和步骤3的代码量也会跟着多起来。 

上述流程存在的问题是:
  * 问题1: 全图转换静态图有时存在较高的使用门槛(学习成本）-> 到底有没有必要将整个模型都转为静态图？
  * 问题2: 尽管全图`jit.to_static`基本都可以成功，但是全图`jit.save`在一些复杂模型中会遇到报错，需要修改用户代码才能避免。
  * 问题3: 用户需要开发专门的动转静脚本、静态图推理脚本，学习静态图相关配置(如TensorRT配置等，同样存在一定学习成本)

## 2.动态图&静态图混合推理新模式  
针对上述问题，我们提出了动态图和静态图混合推理的新模式:  
在这种新模式下， 
  *  1、我们对具有繁琐的控制逻辑的部分，或者那些根本没办法动转静的部分都不做动转静操作，而只对模型网络中的核心耗时部分按照静态图进行推理。 
  *  2、用户只需要维护原本的动态图推理脚本，所有的动转静、Pass优化、缓存生成的操作都是隐式进行的，用户并不感知。   

针对本文的示例代码，用户可以使用以下的新模式直接替代步骤2和步骤3中的代码，以达到等价的推理效果。  
```py
    # 用基于装饰器的方式推理，帮助用户省略步骤2和步骤3的代码，达到同等效果。
    mylayer = paddle.incubate.jit.inference(mylayer)
    # 开启TensorRT后端
    # mylayer = paddle.incubate.jit.inference(mylayer, with_trt=True)
    # 进行推理，返回得到结果
    decorator_result = mylayer(x)
```

### 2.1 功能介绍  
  * 1、支持动态图和静态图混合推理，用户只需修改少量代码即可完成模型推理。  
        具体来讲，用户只需要在动态图组网代码中，将需要动转静的部分用`paddle.incubate.jit.inference`包裹起来即可。  
        部分通过装饰器修饰的函数，会在用户调用该函数时，自动将函数内的操作进行动转静，并保存在磁盘中。
  * 2、支持其他推理的参数，用户可以通过关键字的参数传入即可使用其他的推理加速功能。  
        如`with_trt`表示是否开启TensorRT后端，`trt_precision_mode`表示TensorRT后端的精度等诸多参数。  
        可添加的参数列表参考[paddle.incubate.jit.inference的参数列表](#paddleincubatejitinference的参数列表)。  

### 2.2 装饰器使用方式

#### 目前支持两三种使用方式:
*  方式一:  
    `@paddle.incubate.jit.inference()`放在耗时函数上，
    将函数部分做动转静，并在该函数调用部分进行静态图推理，其他部分仍然使用动态图推理。
*  方式二:   
    `mylayer = paddle.incubate.jit.inference(mylayer)`加速整个`Layer`的`forward`函数
    将整个`Layer`的`forward`函数做动转静，`Layer`全部使用静态图推理。
*  方式三:   
    `mylayer.func = paddle.incubate.jit.inference(mylayer.func)`加速类的局部某个函数
    将类的某个函数做动转静，该函数使用静态图推理，其他部分仍然使用动态图推理。

#### 2.2.1 python动态图推理部署用户: 
*   动态图推理时候，当用户意识到某个模块比较费时间，可以将此模块封装成py函数，然后加上装饰器`paddle.incubate.jit.inference()`，即可获得推理加速。   
    例如:在 transformer 架构的模型中，绝大部分的高耗时部分，应该是这样的语句  
    代码1: 

```py
for block in self.blocks:
    x, y = block(x, y, c, mask)
```

*   那么用户可以将上述语句抽象成下面的函数，并加上装饰器`@paddle.incubate.jit.inference`，即可获得推理加速。  
    代码2: 

```py
@paddle.incubate.jit.inference()
def transformer_blocks(self, x, y, c, mask):
    for block in self.blocks:
        x, y = block(x, y, c, mask)
    return x, y
```

*   之后只需要将原动态图推理中的代码1，换成调用`[x,y] = self.transformer_blocks(x, y, c, mask)`即可。

#### 2.2.2 C++等其他用户:  
*    暂不支持  

## 3.动态图&静态图混合推理使用注意事项  
*   `@paddle.incubate.jit.inference()` 仅适用于推理，不能用于训练。  
*   确保要加速的函数是某`nn.Layer`的成员函数。
    - 也就是说，该函数的第一个参数必须是`self`，且`self`是`nn.Layer`的子类。
    - 这部分在代码中已经加好`assert`判断。
*   如果是用`@paddle.incubate.jit.inference()`的方法使用装饰器，则必须保证调用这个函数的类是该进程唯一的实例。
    - 这个是因为导出的静态图是和`self`强绑定的，例如权重。
    - 如果存在`model1`和`model2`是同一个类的实例，应该避免使用`@paddle.incubate.jit.inference()`装饰器的方式。  
    而是用`model1=paddle.incubate.jit.inference()(model1)`， `model2=paddle.incubate.jit.inference(model2)`代替。
*   做好参数准备，确保参数符合要求
    - 输入参数是`paddle.Tensor` ， `list[paddle.Tensor]`则会被当做是静态图的输入参数
    - 如果是其他类型的参数，如`None`，`bool`，动转静态后将会被固定为常量
    - 函数定义的参数里面禁止含有`*args`和`**kwargs`之类的参数 
    - 并且每个参数在该函数的所有次调用的时候类型必须维持不变，也就是说，你如果第一次是`None`，那么你永远都必须是`None`，如果你第一次是个`Tensor`，那么你永远都必须是`Tensor`。  
*   由于转静后的函数输出只能是`Paddle.Tensor`， 这与原动态图的输出可能会有差异，需要用户调整函数输出衔接。
    - 这一点可以参考PaddleMix中`DIT`模型的推理优化部分:
   https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/pipelines/dit/pipeline_dit.py#L195
```py
    samples_out = self.vae.decode(latents)
    if paddle.incubate.jit.is_inference_mode(self.vae.decode):
        # self.vae.decode run in paddle inference.
        samples = samples_out
    else:
        # 原动态图输出
        samples = samples_out.sample
```

*   输入如果是动态shape的话，当输入的维度的某个值第一次发生变化时，会重新做`jit.save`，并将此维度的这个值标记为`None`，表明此维度可变化，当再次变化的时候则无需再做`jit.save`   
*   TODO
    - 尝试自动释放不再需要的显存
    - 根据业务需要，加入更多的推理参数。

## paddle.incubate.jit.inference的参数列表
```py
    def inference(
        function=None,                  # 可调用的动态图函数。它必须是paddle.nn.Layer的成员函数。如果用作装饰器，则被装饰的函数将被解析为此参数。  
        cache_static_model=False,       # 是否使用磁盘中缓存的静态模型。默认为False。当cache_static_model为True时，静态模型将保存在磁盘中，下次调用会直接使用磁盘中的静态模型  
        save_model_dir=None,            # 静态模型保存的目录。默认为none，即默认是~/.cache/paddle/inference_models/。  
        memory_pool_init_size_mb=1000,  # 内存池初始大小，单位MB。默认为1000。  
        precision_mode="float32",       # 精度模式。默认为"float32"。  
        switch_ir_optim=True,           # 是否开启IR优化。默认为True。  
        switch_ir_debug=False,          # 是否开启IR debug。默认为False。  
        enable_cinn=False,              # 是否开启CINN。默认为False。  
        with_trt=False,                 # 是否开启TensorRT。默认为False。  
        trt_precision_mode="float32",   # TensorRT的精度模式。默认为"float32"。  
        trt_use_static=False,           # 是否缓存 TensorRT build 好的engine。默认为False。  
        collect_shape=False,            # 是否收集shape。默认为False。  
        enable_new_ir=False,            # 是否开启new_ir。默认为True。  
        exp_enable_use_cutlass=False,   # 是否开启cutlass。默认为False。  
        delete_pass_lists=None,         # 删除的pass列表。默认为None。  
    ):
```
