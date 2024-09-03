<h1 align="center" id="title">Vision Architectures from Scratch in PyTorch</h1>

<p align="center"><img src="https://socialify.git.ci/protyayofficial/Vision-Architectures/image?description=1&amp;font=Jost&amp;language=1&amp;name=1&amp;pattern=Circuit%20Board&amp;theme=Light" alt="project-image"></p>

<p id="description">This repository contains implementations of popular vision architectures from scratch using PyTorch. Each model is implemented with clean readable and well-documented code making it easy to understand the underlying mechanics of each architecture.</p>

<h2>🚀 Architectures Implemented:</h2>
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
    <tr>
      <td><strong>VGG16</strong></td>
      <td>Introduced deep networks with small (3x3) convolution filters.</td>
      <td><a href="models/vgg16.py">vgg16.py</a></td>
    </tr>
    <tr>
      <td><strong>VGG19</strong></td>
      <td>A deeper version of VGG16 with 19 layers, focusing on simplicity.</td>
      <td><a href="models/vgg19.py">vgg19.py</a></td>
    </tr>
    <tr>
      <td><strong>GoogLeNet</strong></td>
      <td>Introduced the Inception module for multi-scale feature extraction.</td>
      <td><a href="models/googlenet.py">googlenet.py</a></td>
    </tr>
    <tr>
      <td><strong>ResNet18</strong></td>
      <td>Introduced residual learning to ease the training of deep networks.</td>
      <td><a href="models/resnet18.py">resnet18.py</a></td>
    </tr>
    <tr>
      <td><strong>ResNet34</strong></td>
      <td>Scaled-up ResNet18 with 34 layers, balancing depth and performance.</td>
      <td><a href="models/resnet34.py">resnet34.py</a></td>
    </tr>
    <tr>
      <td><strong>ResNet50</strong></td>
      <td>Deeper ResNet with bottleneck layers for efficient training.</td>
      <td><a href="models/resnet50.py">resnet50.py</a></td>
    </tr>
    <tr>
      <td><strong>ResNet101</strong></td>
      <td>Extends ResNet50 to 101 layers for better feature representation.</td>
      <td><a href="models/resnet101.py">resnet101.py</a></td>
    </tr>
    <tr>
      <td><strong>ResNet152</strong></td>
      <td>The deepest ResNet with 152 layers, achieving top accuracy.</td>
      <td><a href="models/resnet152.py">resnet152.py</a></td>
    </tr>
    <tr>
      <td><strong>InceptionV2</strong></td>
      <td>Improved version of GoogLeNet with BatchNorm and efficient factorization.</td>
      <td><a href="models/inceptionv2.py">inceptionv2.py</a></td>
    </tr>
    <tr>
      <td><strong>DenseNet121</strong></td>
      <td>Uses dense connections to encourage feature reuse and improve efficiency.</td>
      <td><a href="models/densenet121.py">densenet121.py</a></td>
    </tr>
    <tr>
      <td><strong>DenseNet169</strong></td>
      <td>Extends DenseNet121 with more layers, enhancing feature reuse.</td>
      <td><a href="models/densenet169.py">densenet169.py</a></td>
    </tr>
    <tr>
      <td><strong>DenseNet201</strong></td>
      <td>A deeper DenseNet variant with 201 layers for improved performance.</td>
      <td><a href="models/densenet201.py">densenet201.py</a></td>
    </tr>
    <tr>
      <td><strong>DenseNet264</strong></td>
      <td>The deepest DenseNet variant, offering powerful feature extraction.</td>
      <td><a href="models/densenet264.py">densenet264.py</a></td>
    </tr>
    <tr>
      <td><strong>Xception</strong></td>
      <td>Extreme version of Inception using depthwise separable convolutions.</td>
      <td><a href="models/xception.py">xception.py</a></td>
    </tr>
    <tr>
      <td><strong>MobileNetV1</strong></td>
      <td>Efficient architecture using depthwise separable convolutions, optimized for mobile and embedded vision applications.</td>
      <td><a href="models/mobilenetv1.py">mobilenet.py</a></td>
    </tr>
    <tr>
      <td><strong>MobileNetV2</strong></td>
      <td>Improved version of MobileNet with an inverted residual structure and linear bottleneck, designed for efficient mobile and embedded vision tasks.</td>
      <td><a href="models/mobilenetv2.py">mobilenetv2.py</a></td>
    </tr>

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
<ul>
    <li><a href="https://pytorch.org/">PyTorch</a></li>
    <li><a href="https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf">AlexNet</a> by Krizhevsky <em>et al.</em></li>
    <li><a href="https://arxiv.org/pdf/1409.1556">VGG</a> by Simonyan <em>et al.</em></li>
    <li><a href="https://arxiv.org/pdf/1512.03385">ResNet</a> by He <em>et al.</em></li>
    <li><a href="https://arxiv.org/pdf/1608.06993">DenseNet</a> by Huang <em>et al.</em></li>
    <li><a href="https://arxiv.org/pdf/1512.00567">InceptionV2/V3</a> by Szegedy <em>et al.</em></li>
    <li><a href="https://arxiv.org/pdf/1610.02357">Xception</a> by Chollet <em>et al.</em></li>
    <li><a href="https://arxiv.org/pdf/1704.04861">MobileNetV1</a> by Howard <em>et al.</em></li>
    <li><a href="https://arxiv.org/pdf/1801.04381">MobileNetV2</a> by Sandler <em>et al.</em></li>
</ul>

<h2>Contact</h2>
<p>If you have any questions or suggestions, feel free to reach out!</p>

<p><strong>Protyay Dey</strong></p>
<ul>
    <li>Email: <a href="mailto:protyayofficial@gmail.com">protyayofficial@gmail.com</a></li>
    <li>LinkedIn: <a href="https://www.linkedin.com/in/protyaydey">protyaydey</a></li>
    <li>GitHub: <a href="https://www.github.com/protyayofficial">protyayofficial</a></li>
    <li>Website: <a href="https://protyayofficial.github.io">protyayofficial.github.io</a></li>
</ul>

