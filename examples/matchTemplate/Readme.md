#### 目标

通过模型方式实现matchTemplate功能，目前主要有两种需要实现：
1 TM_CCOEFF_NORMED，来自有道，输入hw:109x87,模板:80x56
2 TM_SQDIFF，来自大牛儿，输入hw: 160x180,模板:80x80

#### 实现方法
通过match_template算子实现
op_type: bf16
platform: 182x
input_tpye: uint8

1 TM_CCOEFF_NORMED ((109x87 , 80x56) -> 30x32  ->  1)
    该方法关心极大值的位置
    dst = reduce_sum(inp · tpl) / sqrt(reduce_sum(pow(inp, 2)))  match_template_op
    idx = argmax(dst)   argmax_op

    all deepwise: 4.06ms (目前采用的方案)
    broad_cast mul + add: 4.33ms
    模板尺寸过大存在数值溢出问题：
        尝试将输入进行scale操作，根据公式目前很难保证分子、分母同时保证精度
        目前解决方法，将inp与tpl以mean=128进行shift
2 TM_SQDIFF，来自大牛儿，输入hw: 160x180,模板:80x80
    该方法关心极小值所在的位置
    dst = 1. / sqrt(reduce_sum(pow(inp - tpl， 2)))  match_template_op
    idx = argmax(dst)  argmax_op

    broad_cast sub + deepwise
    inp      tpl      MT_OUT   argmax  cost
    160x180, 80x80 -> 81x101 ->  1     36.57ms
    109x87 , 80x56 -> 30x32  ->  1     3.39ms
    模板尺寸过大同样存在数值溢出问题（此处为极为不相似，因此未进行特殊处理）
    1. / sqrt(x)存在精度截断问题，当前解决方法对x以scale = 1. / (th * tw * 100)进行处理


#### 进展
在极端情况下溢出问题仍然存在：
TM_CCOEFF_NORMED：在当前模板尺寸下，模板像素值与内容像数值 ∈ randint(200, 255);
在通常情况下能获得正确的索引