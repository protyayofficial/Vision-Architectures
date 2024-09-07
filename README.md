<h1 align="center" id="title">Vision Architectures from Scratch in PyTorch</h1>

<p align="center"><img src="https://socialify.git.ci/protyayofficial/Vision-Architectures/image?description=1&amp;font=Jost&amp;language=1&amp;name=1&amp;pattern=Circuit%20Board&amp;theme=Light" alt="project-image"></p>

<p id="description">This repository contains implementations of popular vision architectures from scratch using PyTorch. Each model is implemented with clean readable and well-documented code making it easy to understand the underlying mechanics of each architecture.</p>

<h2>🚀 Architectures Implemented:</h2>

<h3>2012</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>AlexNet</strong></td>
      <td>Revolutionized deep learning by winning ImageNet 2012.</td>
      <td><a href="models/alexnet.py">alexnet.py</a></td>
    </tr>
  </tbody>
</table>

<h3>2014</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>VGG16</strong></td>
      <td rowspan="2">The VGG models are known for their deep network architecture with small 3x3 convolution filters. VGG16 uses 16 layers and is recognized for its simplicity and effectiveness, while VGG19 extends this to 19 layers for potentially better feature extraction.</td>
      <td><a href="models/vgg16.py">vgg16.py</a></td>
    </tr>
    <tr>
        <td><strong>VGG19</strong></td>
        <td><a href="models/vgg19.py">vgg19.py</a></td>
    </tr>
    <tr>
      <td><strong>GoogLeNet</strong></td>
      <td>Introduced the Inception module for multi-scale feature extraction.</td>
      <td><a href="models/googlenet.py">googlenet.py</a></td>
    </tr>
  </tbody>
</table>

<h3>2015</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>ResNet18</strong></td>
      <td rowspan="5">The ResNet family introduces residual learning to facilitate the training of deep networks. ResNet18 and ResNet34 are shallower models offering a balance between depth and performance, while ResNet50, ResNet101, and ResNet152 progressively increase depth and complexity to improve feature representation and accuracy.</td>
      <td><a href="models/resnet18.py">resnet18.py</a></td>
    </tr>
    <tr>
        <td><strong>ResNet34</strong></td>
        <td><a href="models/resnet34.py">resnet34.py</a></td>
    </tr>
    <tr>
        <td><strong>ResNet50</strong></td>
        <td><a href="models/resnet50.py">resnet50.py</a></td>
    </tr>
    <tr>
        <td><strong>ResNet101</strong></td>
        <td><a href="models/resnet101.py">resnet101.py</a></td>
    </tr>
    <tr>
        <td><strong>ResNet152</strong></td>
        <td><a href="models/resnet152.py">resnet152.py</a></td>
    </tr>
    <tr>
        <td><strong>InceptionV2</strong></td>
        <td>Enhances the original GoogLeNet architecture with BatchNorm and efficient factorization for improved performance and reduced computational cost.</td>
        <td><a href="models/inceptionv2.py">inceptionv2.py</a></td>
    </tr>
  </tbody>
</table>

<h3>2016</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>DenseNet121</strong></td>
      <td rowspan="4">The DenseNet family utilizes dense connections to enhance feature reuse and improve efficiency. DenseNet121 is the baseline model, while DenseNet169, DenseNet201, and DenseNet264 progressively increase in depth, with DenseNet264 being the deepest variant for the most powerful feature extraction.</td>
      <td><a href="models/densenet121.py">densenet121.py</a></td>
    </tr>
    <tr>
        <td><strong>DenseNet169</strong></td>
        <td><a href="models/densenet169.py">densenet169.py</a></td>
    </tr>
    <tr>
        <td><strong>DenseNet201</strong></td>
        <td><a href="models/densenet201.py">densenet201.py</a></td>
    </tr>
    <tr>
        <td><strong>DenseNet264</strong></td>
        <td><a href="models/densenet264.py">densenet264.py</a></td>
    </tr>
  </tbody>
</table>

<h3>2017</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Xception</strong></td>
      <td>Extreme version of Inception using depthwise separable convolutions.</td>
      <td><a href="models/xception.py">xception.py</a></td>
    </tr>
    <tr>
      <td><strong>MobileNetV1</strong></td>
      <td>Efficient architecture using depthwise separable convolutions, optimized for mobile and embedded vision applications.</td>
      <td><a href="models/mobilenetv1.py">mobilenetv1.py</a></td>
    </tr>
  </tbody>
</table>

<h3>2018</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MobileNetV2</strong></td>
      <td>Improved version of MobileNet with an inverted residual structure and linear bottleneck, designed for efficient mobile and embedded vision tasks.</td>
      <td><a href="models/mobilenetv2.py">mobilenetv2.py</a></td>
    </tr>
    <tr>
      <td><strong>MNASNet</strong></td>
      <td>Network architecture optimized through neural architecture search (NAS) for efficient mobile and embedded vision applications.</td>
      <td><a href="models/mnasnet.py">mnasnet.py</a></td>
    </tr>
  </tbody>
</table>

<h3>2019</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MobileNetV3 Small</strong></td>
      <td rowspan="2">The Large variant offers advanced features like hard swish and squeeze-and-excitation blocks for enhanced accuracy, while the Small variant focuses on reducing model size without compromising performance.</td>
      <td><a href="models/mobilenetv3_small.py">mobilenetv3_small.py</a></td>
      </tr>
      <tr>
          <td><strong>MobileNetV3 Large</strong></td>
          <td><a href="models/mobilenetv3_large.py">mobilenetv3_large.py</a></td>
      </tr>
    <tr>
        <td><strong>EfficientNetB0</strong></td>
        <td rowspan="8">The EfficientNet-B series uses a compound scaling method to balance depth, width, and resolution across different models. Each version scales up these dimensions to improve accuracy and handle more complex tasks, while maintaining efficiency. The series ranges from B0 with baseline performance to B7, which provides top-tier performance for high-resolution tasks.</td>
        <td><a href="models/efficientnetb0.py">efficientnetb0.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetB1</strong></td>
          <td><a href="models/efficientnetb1.py">efficientnetb1.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetB2</strong></td>
          <td><a href="models/efficientnetb2.py">efficientnetb2.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetB3</strong></td>
          <td><a href="models/efficientnetb3.py">efficientnetb3.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetB4</strong></td>
          <td><a href="models/efficientnetb4.py">efficientnetb4.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetB5</strong></td>
          <td><a href="models/efficientnetb5.py">efficientnetb5.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetB6</strong></td>
          <td><a href="models/efficientnetb6.py">efficientnetb6.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetB7</strong></td>
          <td><a href="models/efficientnetb7.py">efficientnetb7.py</a></td>
      </tr>
  </tbody>
</table>

<h3>2021</h3>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Description</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
      <tr>
          <td><strong>EfficientNetV2-S</strong></td>
          <td rowspan="3">EfficientNetV2 models, including S, M, and L variants, utilize advanced depthwise separable convolutions and optimized MBConv and Fused MBConv blocks.</td>
          <td><a href="models/efficientnetv2_s.py">efficientnetv2_s.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetV2-M</strong></td>
          <td><a href="models/efficientnetv2_m.py">efficientnetv2_m.py</a></td>
      </tr>
      <tr>
          <td><strong>EfficientNetV2-L</strong></td>
          <td><a href="models/efficientnetv2_l.py">efficientnetv2_l.py</a></td>
      </tr>
  </tbody>
</table>


<table>
  <tbody>
    <tr>
      <td><strong>More to come...</strong></td>
      <td>Stay tuned for additional architectures!</td>
      <td></td>
    </tr>
  </tbody>
</table>




<h2>🍰 Contribution Guidelines:</h2>

<ol>
  <li>Fork the repository. </li>
  <li>Create your feature branch (<code>git checkout -b feature/your-feature</code>). </li>
  <li>Commit your changes (<code>git commit -m 'Add some feature'</code>). </li>
  <li>Push to the branch (<code>git push origin feature/your-feature</code>).</li>
  <li>Open a pull request</li>
</ol>
 
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   PyTorch
*   Shell

<h2>🛡️ License:</h2>
This project is licensed under the MIT License - see the <a href="https://github.com/protyayofficial/Vision-Architectures/blob/main/LICENSE">LICENSE</a> file for details.

<h2>🫡 Acknowledgements:</h2>

<h3>2012</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"><strong>AlexNet</strong></a> by Krizhevsky <em>et al.</em></td>
      <td style="text-align: justify;">We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.</td>
    </tr>
  </tbody>
</table>

<h3>2014</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1409.1556"><strong>VGG</strong></a> by Simonyan <em>et al.</em></td>
      <td style="text-align: justify;">In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3x3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.</td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1409.4842"><strong>GoogLeNet</strong></a> by Szegedy <em>et al.</em></td>
      <td style="text-align: justify;">We propose a deep convolutional neural network architecture codenamed "Inception", which was responsible for setting the new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014). The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant. To optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing. One particular incarnation used in our submission for ILSVRC 2014 is called GoogLeNet, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.</td>
    </tr>
  </tbody>
</table>

<h3>2015</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1512.03385"><strong>ResNet</strong></a> by He <em>et al.</em></td>
      <td style="text-align: justify;">Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.</td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1512.00567"><strong>InceptionV2/V3</strong></a> by Szegedy <em>et al.</em></td>
      <td style="text-align: justify;">Convolutional networks are at the core of most state-of-the-art computer vision solutions for a wide variety of tasks. Since 2014 very deep convolutional networks started to become mainstream, yielding substantial gains in various benchmarks. Although increased model size and computational cost tend to translate to immediate quality gains for most tasks (as long as enough labeled data is provided for training), computational efficiency and low parameter count are still enabling factors for various use cases such as mobile vision and big-data scenarios. Here we explore ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set.</td>
    </tr>
  </tbody>
</table>

<h3>2016</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1608.06993"><strong>DenseNet</strong></a> by Huang <em>et al.</em></td>
      <td style="text-align: justify;">Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L conmodels/efficientnetb5.py
    </tr>
  </tbody>
</table>

<h3>2017</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1610.02357"><strong>Xception</strong></a> by Chollet <em>et al.</em></td>
      <td style="text-align: justify;">We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.</td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1704.04861"><strong>MobileNetV1</strong></a> by Howard <em>et al.</em></td>
      <td style="text-align: justify;">We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.</td>
    </tr>
  </tbody>
</table>

<h3>2018</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1801.04381"><strong>MobileNetV2</strong></a> by Sandler <em>et al.</em></td>
      <td style="text-align: justify;">In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. We also describe efficient ways of applying these mobile models to object detection in a novel framework we call SSDLite. Additionally, we demonstrate how to build mobile semantic segmentation models through a reduced form of DeepLabv3 which we call Mobile DeepLabv3. The MobileNetV2 architecture is based on an inverted residual structure where the input and output of the residual block are thin bottleneck layers opposite to traditional residual models which use expanded representations in the input an MobileNetV2 uses lightweight depthwise convolutions to filter features in the intermediate expansion layer. Additionally, we find that it is important to remove non-linearities in the narrow layers in order to maintain representational power. We demonstrate that this improves performance and provide an intuition that led to this design. Finally, our approach allows decoupling of the input/output domains from the expressiveness of the transformation, which provides a convenient framework for further analysis. We measure our performance on Imagenet classification, COCO object detection, VOC image segmentation. We evaluate the trade-offs between accuracy, and number of operations measured by multiply-adds (MAdd), as well as the number of parameters</td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1807.11626"><strong>MnasNet</strong></a> by Tan <em>et al.</em></td>
      <td style="text-align: justify;">Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this paper, we propose an automated mobile neural architecture search (MNAS) approach, which explicitly incorporate model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. Unlike previous work, where latency is considered via another, often inaccurate proxy (e.g., FLOPS), our approach directly measures real-world inference latency by executing the model on mobile phones. To further strike the right balance between flexibility and search space size, we propose a novel factorized hierarchical search space that encourages layer diversity throughout the network. Experimental results show that our approach consistently outperforms state-of-the-art mobile CNN models across multiple vision tasks. On the ImageNet classification task, our MnasNet achieves 75.2% top-1 accuracy with 78ms latency on a Pixel phone, which is 1.8x faster than MobileNetV2 [29] with 0.5% higher accuracy and 2.3x faster than NASNet [36] with 1.2% higher accuracy. Our MnasNet also achieves better mAP quality than MobileNets for COCO object detection.</td>
    </tr>
  </tbody>
</table>

<h3>2019</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/1905.02244"><strong>MobileNetV3</strong></a> by Howard <em>et al.</em></td>
      <td style="text-align: justify;">We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design. MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and then subsequently improved through novel architecture advances. This paper starts the exploration of how automated search algorithms and network design can work together to harness complementary approaches improving the overall state of the art. Through this process we create two new MobileNet models for release: MobileNetV3-Large and MobileNetV3-Small which are targeted for high and low resource use cases. These models are then adapted and applied to the tasks of object detection and semantic segmentation. For the task of semantic segmentation (or any dense pixel prediction), we propose a new efficient segmentation decoder Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP). We achieve new state of the art results for mobile classification, detection and segmentation. MobileNetV3-Large is 3.2\% more accurate on ImageNet classification while reducing latency by 15\% compared to MobileNetV2. MobileNetV3-Small is 4.6\% more accurate while reducing latency by 5\% compared to MobileNetV2. MobileNetV3-Large detection is 25\% faster at roughly the same accuracy as MobileNetV2 on COCO detection. MobileNetV3-Large LR-ASPP is 30\% faster than MobileNetV2 R-ASPP at similar accuracy for Cityscapes segmentation.</td>
    </tr>
    <tr>
      <td><a href="https://arxiv.org/pdf/1905.11946"><strong>EfficientNet</strong></a> by Tan <em>et al.</em></td>
      <td style="text-align: justify;">Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. </td>
    </tr>
  </tbody>
</table>

<h3>2019</h3>
<table>
  <thead>
    <tr>
      <th>Paper</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://arxiv.org/pdf/2104.00298"><strong>EfficientNetV2</strong></a> by Tan <em>et al.</em></td>
      <td style="text-align: justify;">This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop this family of models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller. Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose to adaptively adjust regularization (e.g., dropout and data augmentation) as well, such that we can achieve both fast training and good accuracy. With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources. </td>
    </tr>
  </tbody>
</table>

<h2>Contact</h2>
<p>If you have any questions or suggestions, feel free to reach out!</p>

<p><strong>Protyay Dey</strong></p>
<ul>
    <li>Email: <a href="mailto:protyayofficial@gmail.com">protyayofficial@gmail.com</a></li>
    <li>LinkedIn: <a href="https://www.linkedin.com/in/protyaydey">protyaydey</a></li>
    <li>GitHub: <a href="https://www.github.com/protyayofficial">protyayofficial</a></li>
    <li>Website: <a href="https://protyayofficial.github.io">protyayofficial.github.io</a></li>
</ul>

