#### 目标

通过模型方式实现matchTemplate功能，目前主要有两种需要实现：
1 TM_CCOEFF_NORMED，来自有道，输入hw:109x87,模板:80x56
2 TM_SQDIFF，来自大牛儿，输入hw: 160x180,模板:80x80

#### 实现方法

通过match_template算子实现，尤其第一步先把输入展开，用FC实现或者广播乘法实现卷积的功能
（不直接用卷积是因为kernel太大）

#### 进展

目前只粗略的实现了ccoeff_normed功能，还有比较大的优化空间，模型推理时间需要控制在5ms以内。

待完成（by charle.hu)