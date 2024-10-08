<h1 style="align: center">🧠 Vision Architectures from Scratch in PyTorch</h1>

<img src="https://socialify.git.ci/protyayofficial/Vision-Architectures/image?description=1&font=Jost&language=1&name=1&pattern=Circuit%20Board&theme=Light" alt="project-image" class="project-image">

<br>
  
<div class="section">
    <h2>🎯 Project Description</h2>
    <p>This repository contains implementations of popular vision architectures from scratch using PyTorch. Each model is implemented with clean, readable, and well-documented code, making it easy to understand the underlying mechanics of each architecture.</p>
</div>

<div class="section">
  <h2>🚀 Image Classification Architectures</h2>
  
  <h3>1. Convolutional Neural Networks (CNNs)</h3>
  <h4>2012</h4>
  <table>
      <tr>
          <th>Model</th>
          <th>Abstract</th>
          <th>Link</th>
      </tr>
      <tr>
          <td><a href="https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"><strong>AlexNet</strong></a> by Krizhevsky <em>et al.</em></td>
          <td>
              <div class="abstract">We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art...</div>
          </td>
          <td><a href="models/alexnet.py">alexnet.py</a></td>
      </tr>
  </table>
<h4>2014</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="https://arxiv.org/pdf/1409.1556"><strong>VGG</strong></a> by Simonyan <em>et al.</em></td>
      <td rowspan="2" style="text-align: justify;">In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.</td>
      <td><a href="models/vgg16.py">vgg16.py</a></td>
    </tr>
    <tr>
        <td><a href="models/vgg19.py">vgg19.py</a></td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1409.4842"><strong>GoogLeNet</strong></a> by Szegedy <em>et al.</em></td>
      <td style="text-align: justify;">We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.</td>
      <td><a href="models/googlenet.py">googlenet.py</a></td>
    </tr>
  </tbody>
</table>

<h4>2015</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><a href="https://arxiv.org/pdf/1512.03385"><strong>ResNet</strong></a> by He <em>et al.</em></td>
      <td rowspan="5" style="text-align: justify;">Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.</td>
      <td><a href="models/resnet18.py">resnet18.py</a></td>
    </tr>
    <tr>
        <td><a href="models/resnet34.py">resnet34.py</a></td>
    </tr>
    <tr>
        <td><a href="models/resnet50.py">resnet50.py</a></td>
    </tr>
    <tr>
        <td><a href="models/resnet101.py">resnet101.py</a></td>
    </tr>
    <tr>
        <td><a href="models/resnet152.py">resnet152.py</a></td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/pdf/1512.00567"><strong>InceptionV2/V3</strong></a> by Szegedy <em>et al.</em></td>
        <td style="text-align: justify;">Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set.</td>
        <td><a href="models/inceptionv2.py">inceptionv2.py</a></td>
    </tr>
  </tbody>
</table>

<h4>2016</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><a href="https://arxiv.org/pdf/1608.06993"><strong>DenseNet</strong></a> by Huang <em>et al.</em></td>
      <td rowspan="4" style="text-align: justify">Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. </td>
      <td><a href="models/densenet121.py">densenet121.py</a></td>
    </tr>
    <tr>
        <td><a href="models/densenet169.py">densenet169.py</a></td>
    </tr>
    <tr>
        <td><a href="models/densenet201.py">densenet201.py</a></td>
    </tr>
    <tr>
        <td><a href="models/densenet264.py">densenet264.py</a></td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1602.07360"><strong>SqueezeNet</strong></a> by Iandola <em>et al.</em></td>
      <td style="text-align: justify">Recent research on deep neural networks has focused primarily on improving accuracy. For a given accuracy level, it is typically possible to identify multiple DNN architectures that achieve that accuracy level. With equivalent accuracy, smaller DNN architectures offer at least three advantages: (1) Smaller DNNs require less communication across servers during distributed training. (2) Smaller DNNs require less bandwidth to export a new model from the cloud to an autonomous car. (3) Smaller DNNs are more feasible to deploy on FPGAs and other hardware with limited memory. To provide all of these advantages, we propose a small DNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet).</td>
      <td><a href="models/squeezenet.py">squeezenet.py</a></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://arxiv.org/pdf/1611.05431"><strong>ResNeXt</strong></a> by Xie <em>et al.</em></td>
        <td rowspan="2" style="text-align: justify;">We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call "cardinality" (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online.</td>
        <td><a href="models/resnext50.py">resnext50.py</a></td>
    </tr>
    <tr>
      <td><a href="models/resnext101.py">resnext101.py</a></td>
    </tr>
  </tbody>
</table>

<h4>2017</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1610.02357"><strong>Xception</strong></a> by Chollet <em>et al.</em></td>
      <td style="text-align: justify;">We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.</td>
      <td><a href="models/xception.py">xception.py</a></td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1704.04861"><strong>MobileNetV1</strong></a> by Howard <em>et al.</em></td>
      <td style="text-align: justify;">We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.</td>
      <td><a href="models/mobilenetv1.py">mobilenetv1.py</a></td>
    </tr>
  </tbody>
</table>

<h4>2018</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1801.04381"><strong>MobileNetV2</strong></a> by Sandler <em>et al.</em></td>
      <td style="text-align: justify;">In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters</td>
      <td><a href="models/mobilenetv2.py">mobilenetv2.py</a></td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1807.11626"><strong>MnasNet</strong></a> by Tan <em>et al.</em></td>
      <td style="text-align: justify;">Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this paper, we propose an automated mobile neural architecture search (MNAS) approach, which explicitly incorporate model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. Unlike previous work, where latency is considered via another, often inaccurate proxy (e.g., FLOPS), our approach directly measures real-world inference latency by executing the model on mobile phones. To further strike the right balance between flexibility and search space size, we propose a novel factorized hierarchical search space that encourages layer diversity throughout the network. Experimental results show that our approach consistently outperforms state-of-the-art mobile CNN models across multiple vision tasks. On the ImageNet classification task, our MnasNet achieves 75.2% top-1 accuracy with 78ms latency on a Pixel phone, which is 1.8x faster than MobileNetV2 [29] with 0.5% higher accuracy and 2.3x faster than NASNet [36] with 1.2% higher accuracy. Our MnasNet also achieves better mAP quality than MobileNets for COCO object detection.</td>
      <td><a href="models/mnasnet.py">mnasnet.py</a></td>
    </tr>
  </tbody>
</table>

<h4>2019</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><a href="https://arxiv.org/pdf/1905.02244"><strong>MobileNetV3</strong></a> by Howard <em>et al.</em></td>
      <td rowspan="3" style="text-align: justify;">We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2\% more accurate on ImageNet classification while reducing latency by 15\% compared to MobileNetV2. MobileNetV3-Small is 4.6\% more accurate while reducing latency by 5\% compared to MobileNetV2. MobileNetV3-Large detection is 25\% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30\% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.</td>
    </tr>
      <td><a href="models/mobilenetv3_small.py">mobilenetv3_small.py</a></td>
      </tr>
      <tr>
          <td><a href="models/mobilenetv3_large.py">mobilenetv3_large.py</a></td>
      </tr>
    <tr>
        <td rowspan="8"><a href="https://arxiv.org/pdf/1905.11946"><strong>EfficientNet</strong></a> by Tan <em>et al.</em></td>
      <td rowspan="8" style="text-align: justify;">Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.</td>
        <td><a href="models/efficientnetb0.py">efficientnetb0.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetb1.py">efficientnetb1.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetb2.py">efficientnetb2.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetb3.py">efficientnetb3.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetb4.py">efficientnetb4.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetb5.py">efficientnetb5.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetb6.py">efficientnetb6.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetb7.py">efficientnetb7.py</a></td>
      </tr>
  </tbody>
</table>

<h4>2021</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
      <tr>
          <td rowspan="3"><a href="https://arxiv.org/pdf/2104.00298"><strong>EfficientNetV2</strong></a> by Tan <em>et al.</em></td>
          <td rowspan="3" style="text-align: justify;">This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller. Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout and data augmentation) as well, such that we can achieve both fast training and good accuracy. With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources.</td>
          <td><a href="models/efficientnetv2_s.py">efficientnetv2_s.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetv2_m.py">efficientnetv2_m.py</a></td>
      </tr>
      <tr>
          <td><a href="models/efficientnetv2_l.py">efficientnetv2_l.py</a></td>
      </tr>
  </tbody>
  <!-- <tbody>
      <tr>
        <td rowspan="5"><a href=""><strong>CoATNet</strong></a> by Dai <em>et al.</em></td>
        <td rowspan="5" style="text-align: justify;"> Transformers have attracted increasing interests in computer vision, but they still fall behind state-of-the-art convolutional networks. In this work, we show that while Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. To effectively combine the strengths from both architectures, we present CoAtNets(pronounced "coat" nets), a family of hybrid models built from two key insights: (1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention; (2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency. Experiments show that our CoAtNets achieve state-of-the-art performance under different resource constraints across various datasets: Without extra data, CoAtNet achieves 86.0% ImageNet top-1 accuracy; When pre-trained with 13M images from ImageNet-21K, our CoAtNet achieves 88.56% top-1 accuracy, matching ViT-huge pre-trained with 300M images from JFT-300M while using 23x less data; Notably, when we further scale up CoAtNet with JFT-3B, it achieves 90.88% top-1 accuracy on ImageNet, establishing a new state-of-the-art result. </td>
        <td><a href="models/CoATNet0.py">CoATNet0.py</a></td>
      </tr>
      <tr>
        <td><a href="models/CoATNet1.py">CoATNet1.py</a></td>
      </tr>
      <tr>
        <td><a href="models/CoATNet2.py">CoATNet2.py</a></td>
      </tr>
      <tr>
        <td><a href="models/CoATNet3.py">CoATNet3.py</a></td>
      </tr>
      <tr>
        <td><a href="models/CoATNet4.py">CoATNet4.py</a></td>
      </tr>
  </tbody> -->
</table>

<h4>2022</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
      <tr>
        <td rowspan="5"><a href=""><strong>ConvNeXt</strong></a> by Liu <em>et al.</em></td>
        <td rowspan="5" style="text-align: justify;"> The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets. </td>
        <td><a href="models/ConvNeXtT.py">ConvNeXtT.py</a></td>
      </tr>
      <tr>
        <td><a href="models/ConvNeXtS.py">ConvNeXtS.py</a></td>
      </tr>
      <tr>
        <td><a href="models/ConvNeXtB.py">ConvNeXtB.py</a></td>
      </tr>
      <tr>
        <td><a href="models/ConvNeXtL.py">ConvNeXtL.py</a></td>
      </tr>
      <tr>
        <td><a href="models/ConvNeXtXL.py">ConvNeXtXL.py</a></td>
      </tr>
  </tbody>
</table>

<h3> 2. Attention-based Architectures </h2>

<h4>2017</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
      <tr>
          <td rowspan="2"><a href="https://arxiv.org/pdf/1704.06904"><strong>Residual Attention Network</strong></a> by Wang <em>et al.</em></td>
          <td rowspan="2" style="text-align: justify;">In this work, we propose "Residual Attention Network", a convolutional neural network using attention mechanism which can incorporate with state-of-art feed forward network architecture in an end-to-end training fashion. Our Residual Attention Network is built by stacking Attention Modules which generate attention-aware features. The attention-aware features from different modules change adaptively as layers going deeper. Inside each Attention Module, bottom-up top-down feedforward structure is used to unfold the feedforward and feedback attention process into a single feedforward process. Importantly, we propose attention residual learning to train very deep Residual Attention Networks which can be easily scaled up to hundreds of layers. Extensive analyses are conducted on CIFAR-10 and CIFAR-100 datasets to verify the effectiveness of every module mentioned above. Our Residual Attention Network achieves state-of-the-art object recognition performance on three benchmark datasets including CIFAR-10 (3.90% error), CIFAR-100 (20.45% error) and ImageNet (4.8% single model and single crop, top-5 error). Note that, our method achieves 0.6% top-1 accuracy improvement with 46% trunk depth and 69% forward FLOPs comparing to ResNet-200. The experiment also demonstrates that our network is robust against noisy labels.
          </td>
          <td><a href="models/RAN56.py">RAN56.py</a></td>
      </tr>
      <tr>
          <td><a href="models/RAN92.py">RAN92.py</a></td>
      </tr>
      <tr>
      <td rowspan="2"><a href="https://arxiv.org/pdf/1709.01507"><strong>SENet</strong></a> by Hu <em>et al.</em></td>
      <td rowspan="2" style="text-align: justify;">The central building block of convolutional neural networks (CNNs) is the convolution operator, which enables networks to construct informative features by fusing both spatial and channel-wise information within local receptive fields at each layer. A broad range of prior research has investigated the spatial component of this relationship, seeking to strengthen the representational power of a CNN by enhancing the quality of spatial encodings throughout its feature hierarchy. In this work, we focus instead on the channel relationship and propose a novel architectural unit, which we term the "Squeeze-and-Excitation" (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. We show that these blocks can be stacked together to form SENet architectures that generalise extremely effectively across different datasets. We further demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at slight additional computational cost. Squeeze-and-Excitation Networks formed the foundation of our ILSVRC 2017 classification submission which won first place and reduced the top-5 error to 2.251%, surpassing the winning entry of 2016 by a relative improvement of ~25%.</td>
      <td><a href="models/seresnet50.py">seresnet50.py</a></td>
    </tr>
    <tr>
      <td><a href="models/seresnext50.py">seresnext50.py</a></td>
    </tr>
  </tbody>
</table>

<h4>2018</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
      <tr>
          <td><a href="https://arxiv.org/pdf/1704.06904"><strong>CBAM</strong></a> by Woo <em>et al.</em></td>
          <td style="text-align: justify;">We propose Convolutional Block Attention Module (CBAM), a simple yet effective attention module for feed-forward convolutional neural networks. Given an intermediate feature map, our module sequentially infers attention maps along two separate dimensions, channel and spatial, then the attention maps are multiplied to the input feature map for adaptive feature refinement. Because CBAM is a lightweight and general module, it can be integrated into any CNN architectures seamlessly with negligible overheads and is end-to-end trainable along with base CNNs. We validate our CBAM through extensive experiments on ImageNet-1K, MS~COCO detection, and VOC~2007 detection datasets. Our experiments show consistent improvements in classification and detection performances with various models, demonstrating the wide applicability of CBAM. </td>
          <td><a href="models/cbamresnet50.py">cbamresnet50.py</a></td>
      </tr>
  </tbody>
</table>
</div>

<h3> 3. Transformer-based Architectures </h2>

<h4>2021</h4>
<table>
  <thead>
    <tr>
      <th style="text-align: center">Model</th>
      <th style="text-align: center">Abstract</th>
      <th style="text-align: center">Link</th>
    </tr>
  </thead>
  <tbody>
      <tr>
        <td rowspan="3"><a href="https://arxiv.org/pdf/2010.11929"><strong>ViT</strong></a> by Dosovitskiy <em>et al.</em></td>
        <td rowspan="3" style="text-align: justify;">While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
        </td>
        <td><a href="models/ViTBase.py">ViTBase.py</a></td>
      </tr>
      <tr>
        <td><a href="models/ViTLarge.py">ViTLarge.py</a></td>
      </tr>
      <tr>
        <td><a href="models/ViTHuge.py">ViTHuge.py</a></td>
      </tr>
  </tbody>
  <tbody>
      <tr>
        <td rowspan="3"><a href="https://arxiv.org/pdf/2103.14030"><strong>Swin Transformer</strong></a> by Liu <em>et al.</em></td>
        <td rowspan="3" style="text-align: justify;">This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures.
        </td>
        <td><a href="models/SwinT.py">SwinT.py</a></td>
      </tr>
      <tr>
        <td><a href="models/SwinS.py">SwinS.py</a></td>
      </tr>
      <tr>
        <td><a href="models/SwinB.py">SwinB.py</a></td>
      </tr>
  </tbody>
</table>
</div>

<div class="section">
  <h2>🚀 Image Segmentation Architectures</h2>
  <h4>2015</h4>
  <table>
    <thead>
      <tr>
        <th style="text-align: center">Model</th>
        <th style="text-align: center">Abstract</th>
        <th style="text-align: center">Link</th>
      </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href=""><strong>UNet</strong></a> by Ronneberger <em>et al.</em></td>
            <td style="text-align: justify;"> There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU.  </td>
            <td><a href="models/UNet.py">UNet.py</a></td>
        </tr>
    </tbody>
  </table>
</div>

<div class="section">
  <h2>🍰 Contribution Guidelines</h2>
  <ol>
      <li>Fork the repository</li>
      <li>Create your feature branch (<code>git checkout -b feature/your-feature</code>)</li>
      <li>Commit your changes (<code>git commit -m 'Add some feature'</code>)</li>
      <li>Push to the branch (<code>git push origin feature/your-feature</code>)</li>
      <li>Open a pull request</li>
  </ol>
</div>
<div class="section">
  <h2>💻 Built with</h2>
  <ul>
      <li>Python</li>
      <li>PyTorch</li>
  </ul>
</div>
    
<div class="section">
  <h2>🛡️ License</h2>
  <p>This project is licensed under the MIT License - see the <a href="https://github.com/protyayofficial/Vision-Architectures/blob/main/LICENSE">LICENSE</a> file for details.</p>
</div>
    
<div class="section">
  <h2>🫡 Acknowledgements</h2>
  <p>I would like to express my deepest gratitude to the incredible research community whose groundbreaking work laid the foundation for this project. The implementation of these vision classification architectures would not have been possible without the invaluable contributions of the authors and researchers who published the original papers, which continue to inspire innovation in the field of computer vision. I extend my thanks to the following for their pioneering work. Also would like to thank PyTorch for providing a powerful and flexible deep learning framework. Lastly, a special thank you to the broader machine learning community for consistently pushing the boundaries of what's possible, inspiring me to tackle this project with creativity and determination. This repository is a testament to the collaborative spirit of research, and I hope it will contribute to future advancements in computer vision.</p>
</div>

<div class="section">
  <h2>📞 Contact</h2>
  <p>If you have any questions or suggestions, feel free to reach out!</p>
  <p><strong>Protyay Dey</strong></p>
  <ul>
      <li>📧 Email: <a href="mailto:protyayofficial@gmail.com">protyayofficial@gmail.com</a></li>
      <li>💼 LinkedIn: <a href="https://www.linkedin.com/in/protyaydey">protyaydey</a></li>
      <li>👨‍💻 GitHub: <a href="https://www.github.com/protyayofficial">protyayofficial</a></li>
      <li>🌐 Website: <a href="https://protyayofficial.github.io">protyayofficial.github.io</a></li>
  </ul>
</div>
