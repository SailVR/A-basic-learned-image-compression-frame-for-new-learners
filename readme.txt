models文件夹下：
命名都很通俗易懂，不解释了

如何跑：
python3 -m models.analysis
就可以执行analysis.py 里面的'__main__'了

test.py 以dataloader的形式进行测试
test_img.py 以单个img送入的形式进行测试

model.py
模型为ImageCompressor，直接python3 model.py就可以查看是否能运行

train.py 训练
解释：
主要修改地方：
tot_step：迭代次数
decay_interval：到这里学习率筛检为1e-5
save_model_freq：每多少次进行一次验证模型

至少tot_epoch的涉及epoch无需修改，默认很大就行了，训练是按照迭代次数来的。

对于parser：
-n：保存的名字，即checkpoint文件夹下的子文件夹
-p：加载的权重是啥，没有则从头训练，有则加载这个权重
--seed：不用管
--train：训练的文件夹
--val：验证的文件夹（直接kodak24）

如果model.py可以直接跑通，那么就可以直接跑train_f.py
建议运行命令为 python3 train.py -n "你想要的命名"

test.py
只需要修改parser里面的
-p：指定好权重
--val：测试的数据
