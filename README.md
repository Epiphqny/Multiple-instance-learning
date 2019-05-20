# Multiple-instance-learning

Pytorch implementation of three Multiple Instance Learning or Multi-classification papers

三种多示例学习方法实现，用于图像的多标签，其中 visual_concept效果最好

* data_process: 构造词汇数据词典，三个方法均通用
* CNN-RNN: A Unified Framework for Multi-label Image Classification https://arxiv.org/abs/1604.04573
* Visual_concept: From captions to visual concepts and back https://arxiv.org/abs/1411.4952?context=cs 
* DeepMIML: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/aaai17deepMIML.pdf

## Data prepare

We will not provide the original dataset, but you can build it using your own dataset. Among them, **resized2014** is image dataset, **img_tag.txt** is the mapping dict file of image to tags, having that, you can generate the **zh_vocab.pkl** vocabulary file using https://github.com/Epiphqny/Multiple-instance-learning/blob/master/data_process/build_vocab.py

### Examples

img_tag.txt(with number id represent different image name):

1\tab girl,bottle,car

2\tab boy

3\tab child,bike

...

zh_vocab.pkl:

self.idx2word={1:girl,2:bottle,3:boy,4:car...}

self.word2idx={girl:1,bottle:2,boy:3,car:4...}


