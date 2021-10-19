## Vision-Transformer-CIFAR10

在此项目中，我们集合了当前一些vision transformer的Pytorch实现，并尝试在CIFAR-10数据集训练。

## 使用

### 1. 训练

```bash
python train.py-net vit -gpu
```

### 2. 验证

```bash
python test.py -net vit -weights path_to_the_weight
```

## 结果

Transformer缺少CNN的归纳偏置，通常需要大量训练数据和数据增强才能达到良好的效果。当前实现还未对数据增强以及学习率等超参进一步微调。

## 笔记

下面是参照别人的实现中的一些记录。[笔记](vit_notes.md ':include')

## 已完成

- [x] [ViT]: https://arxiv.org/abs/2010.11929

- [x] [DeiT]: https://arxiv.org/abs/2012.12877

- [x] [DeepViT]: https://arxiv.org/abs/2103.11886

- [x] [CaiT]: https://arxiv.org/abs/2103.17239

- [x] [CeiT]: https://arxiv.org/abs/2103.11816

- [x] [CPVT]:https://arxiv.org/abs/2102.10882

- [x] [CvT]: https://arxiv.org/abs/2103.15808

- [x] [LeViT]: https://arxiv.org/abs/2104.01136

- [x] [PVT]:https://arxiv.org/abs/2102.12122

- [x] [PVTv2]:https://arxiv.org/abs/2106.13797

- [x] [Swin]: https://arxiv.org/abs/2103.14030

- [x] [Shuffle]:https://arxiv.org/abs/2106.03650

## TODO

- [ ] 数据增强（cutmix，mixup等）

- [ ] 超参调整

- [ ] [PiT]: https://arxiv.org/abs/2103.16302

- [ ] [Tokens-to-Token]: https://arxiv.org/abs/2101.11986

- [ ] [CrossViT]: https://arxiv.org/abs/2103.14899

- [ ] [LocalViT]: https://arxiv.org/abs/2104.05707
- [ ] [Twins]: https://arxiv.org/abs/2104.13840


## 参考

https://github.com/lucidrains/vit-pytorch

https://github.com/weiaicunzai/pytorch-cifar100

https://github.com/berniwal/swin-transformer-pytorch

https://github.com/whai362/PVT