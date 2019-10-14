import TensorFlow
import LayersDataFormat

@differentiable
func mish<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    return input * tanh(softplus(input))
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
    @differentiable
    func l2Loss() -> Tensor<Scalar> {
        return squared().sum()
    }
    
    @differentiable(wrt: self)
    func l2Norm(alongAxes axes: [Int]) -> Tensor<Scalar> {
        return TensorFlow.sqrt(squared().sum(alongAxes: axes))
    }
    
    @differentiable(wrt: self)
    func weightNormalized() -> Tensor<Scalar> {
        let axes = Array<Int>(shape.indices.dropLast())
        let centered = self - self.mean(alongAxes: axes)
        return centered / (centered.l2Norm(alongAxes: axes))
    }
}

func makeStrides(stride: Int, dataFormat: Raw.DataFormat) -> (Int, Int, Int, Int) {
    let strides: (Int, Int, Int, Int)
    switch dataFormat {
    case .nchw:
        strides = (1, 1, stride, stride)
    case .nhwc:
        strides = (1, stride, stride, 1)
    }
    return strides
}

struct WeightNormConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    var filter: Tensor<Scalar> {
        didSet { filter = filter.weightNormalized() }
    }

    var g: Tensor<Scalar>
    @noDerivative var stride: Int = 1
    @noDerivative var dataFormat: Raw.DataFormat = .nhwc
    /*var differentiableVectorView: TangentVector {
        get { TangentVector(filter: filter, g: g) }
        set { filter = newValue.filter; g = newValue.g }
    }*/
    init(filter: Tensor<Scalar>,
         g: Tensor<Scalar>,
         stride: Int = 1,
         dataFormat: Raw.DataFormat = .nhwc)
    {
        self.filter = filter
        self.g = g
        self.stride = stride
        self.dataFormat = dataFormat
        defer {
            self.filter = self.filter
        }
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>{
        return input.convolved2DDF(withFilter: filter * g,
                                   strides: makeStrides(stride: stride, dataFormat: dataFormat),
                                   padding: .same,
                                   dataFormat: dataFormat)
    }
    
    mutating func replaceParameters(_ newValue: TangentVector) {
        filter = newValue.filter
        g = newValue.g
    }
}

struct WeightNormDense<Scalar: TensorFlowFloatingPoint>: Layer {
    var weight: Tensor<Scalar> {
        didSet { weight = weight.weightNormalized() }
    }
    var bias, g: Tensor<Scalar>
    /*var differentiableVectorView: TangentVector {
        get { TangentVector(weight: weight, bias: bias, g: g) }
        set { weight = newValue.weight; bias = newValue.bias; g = newValue.g }
    }*/
    init(weight: Tensor<Scalar>, bias: Tensor<Scalar>, g: Tensor<Scalar>) {
        self.weight = weight
        self.bias = bias
        self.g = g
        defer {
            self.weight = self.weight
        }
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return matmul(input + bias, weight * g) //weight.weightNormalized(g: g))
    }
    
    mutating func replaceParameters(_ newValue: TangentVector) {
        weight = newValue.weight
        bias = newValue.bias
        g = newValue.g
    }
}

struct PreactConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    var filter: Tensor<Scalar> {
        didSet { filter = filter.weightNormalized() }
    }

    var bias1, bias2, g: Tensor<Scalar>
    @noDerivative let stride: Int
    @noDerivative var dataFormat: Raw.DataFormat = .nhwc
    /*var differentiableVectorView: TangentVector {
        get { TangentVector(filter: filter, bias1: bias1, bias2: bias2, g: g) }
        set {
            filter = newValue.filter
            bias1 = newValue.bias1
            bias2 = newValue.bias2
            g = newValue.g
        }
    }*/
    init(filter: Tensor<Scalar>,
         bias1: Tensor<Scalar>,
         bias2: Tensor<Scalar>,
         g: Tensor<Scalar>,
         stride: Int = 1,
         dataFormat: Raw.DataFormat = .nhwc)
    {
        self.filter = filter
        self.bias1 = bias1
        self.bias2 = bias2
        self.g = g
        self.stride = stride
        self.dataFormat = dataFormat
        defer {
            self.filter = filter
        }
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let tmp = mish(input + bias1) + bias2
        return tmp.convolved2DDF(withFilter: filter * g,
                                 strides: makeStrides(stride: stride, dataFormat: dataFormat),
                                 padding: .same,
                                 dataFormat: dataFormat)
    }
    
    mutating func replaceParameters(_ newValue: TangentVector) {
        filter = newValue.filter
        bias1 = newValue.bias1
        bias2 = newValue.bias2
        g = newValue.g
    }
}

struct Shortcut<Scalar: TensorFlowFloatingPoint>: Differentiable {
    @noDerivative let shortcutOp: @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    
    init(stride: Int = 1, featureIncrease: Int = 0, dataFormat: Raw.DataFormat = .nhwc) {
        if stride > 1 || featureIncrease != 0 {
            let channelAxis = dataFormat == .nchw ? 1 : 3
            var padding = [(before: 0, after: 0),
                           (before: 0, after: 0),
                           (before: 0, after: 0),
                           (before: 0, after: 0)]
            padding[channelAxis] = (before: 0, after: featureIncrease)
            
            self.shortcutOp = {input in
                let strides = makeStrides(stride: stride, dataFormat: dataFormat)
                let tmp = input.averagePooledDF(kernelSize: strides,
                                                strides: strides,
                                                padding: .same,
                                                dataFormat: dataFormat)
                return tmp.padded(forSizes: padding)
                }
        } else {
            self.shortcutOp = identity
        }
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return shortcutOp(input)
    }
}

struct PreactResidualBlock<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative let stride: Int
    @noDerivative let featureIn: Int
    @noDerivative let featureOut: Int
    @noDerivative let shortcut: Shortcut<Scalar>
    var conv1: PreactConv2D<Scalar>
    var conv2: PreactConv2D<Scalar>
    var multiplier = Tensor<Scalar>(ones: [1,1,1,1])
    var bias = Tensor<Scalar>(zeros: [1,1,1,1])
    /*var differentiableVectorView: TangentVector {
        get { TangentVector(//shortcut: shortcut.differentiableVectorView,
                            conv1: conv1.differentiableVectorView,
                            conv2: conv2.differentiableVectorView,
                            multiplier: multiplier,
                            bias: bias)
            }
        set { //shortcut.differentiableVectorView = newValue.shortcut
              conv1.differentiableVectorView = newValue.conv1
              conv2.differentiableVectorView = newValue.conv2
              multiplier = newValue.multiplier
              bias = newValue.bias
            }
    }*/

    public init(
        featureIn: Int,
        featureOut: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        dataFormat: Raw.DataFormat = .nhwc
    ) {
        self.stride = stride
        self.featureIn = featureIn
        self.featureOut = featureOut
        self.shortcut = Shortcut(stride: stride,
                                 featureIncrease: featureOut - featureIn,
                                 dataFormat: dataFormat)
        self.conv1 = PreactConv2D(
            filter: Tensor(orthogonal: [kernelSize, kernelSize, featureIn, featureOut]),
            bias1: Tensor(zeros: [1,1,1,1]),
            bias2: Tensor(zeros: [1,1,1,1]),
            g: Tensor(ones: [featureOut]) * sqrt(Scalar(2 * featureIn) / Scalar(featureOut)),
            stride: stride,
            dataFormat: dataFormat
        )
        self.conv2 = PreactConv2D(
            filter: Tensor(orthogonal: [kernelSize, kernelSize, featureOut, featureOut]),
            bias1: Tensor(zeros: [1,1,1,1]),
            bias2: Tensor(zeros: [1,1,1,1]),
            g: Tensor(ones: [featureOut]) * rsqrt(Tensor<Scalar>(5)),
            stride: 1,
            dataFormat: dataFormat
        )
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let tmp = conv2(conv1(input))
        let sc = shortcut(input)
        return tmp * multiplier + bias + sc
    }
    
    mutating func replaceParameters(_ newValue: TangentVector) {
        conv1.replaceParameters(newValue.conv1)
        conv2.replaceParameters(newValue.conv2)
        multiplier = newValue.multiplier
        bias = newValue.bias
    }
}

public struct PreactResNet<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative var dataFormat: Raw.DataFormat = .nhwc
    var multiplier1 = Tensor<Scalar>(ones: [1,1,1,1])
    var bias1 = Tensor<Scalar>(zeros: [1,1,1,1])
    var conv1: WeightNormConv2D<Scalar>
    var blocks: [PreactResidualBlock<Scalar>]
    var multiplier2 = Tensor<Scalar>(ones: [1,1,1,1])
    var bias2 = Tensor<Scalar>(zeros: [1,1,1,1])
    var dense1: WeightNormDense<Scalar>
    /*public var differentiableVectorView: TangentVector {
        get { TangentVector(multiplier1: multiplier1,
                            bias1: bias1,
                            conv1: conv1.differentiableVectorView,
                            blocks: blocks.differentiableVectorView,
                            multiplier2: multiplier2,
                            bias2: bias2,
                            dense1: dense1.differentiableVectorView)
            }
        set { multiplier1 = newValue.multiplier1
              conv1.differentiableVectorView = newValue.conv1
              bias1 = newValue.bias1
              for ii in 0..<blocks.count {
                  blocks[ii].differentiableVectorView = newValue.blocks[ii]
              }
              multiplier2 = newValue.multiplier2
              bias2 = newValue.bias2
              dense1.differentiableVectorView = newValue.dense1
            }
    }*/

    public init(dataFormat: Raw.DataFormat = .nhwc) {
        self.dataFormat = dataFormat
        let depth = 16
        let depth2 = 64
        let depth3 = 128
        let depth4 = 256
        let resUnitPerBlock = 5
        self.conv1 = WeightNormConv2D(
            filter: Tensor(orthogonal: [3, 3, 3, depth]),
            g: Tensor(ones: [depth]) * sqrt(Scalar(2 * 3) / Scalar(depth)),
            stride: 1,
            dataFormat: dataFormat
        )
        
        self.blocks = [PreactResidualBlock(featureIn: depth,
                                           featureOut: depth2,
                                           kernelSize: 3,
                                           stride: 1,
                                           dataFormat: dataFormat)]
        for _ in 1 ..< resUnitPerBlock {
            self.blocks += [PreactResidualBlock(featureIn: depth2,
                                                featureOut: depth2,
                                                kernelSize: 3,
                                                stride: 1,
                                                dataFormat: dataFormat)]
        }
        self.blocks += [PreactResidualBlock(featureIn: depth2,
                                            featureOut: depth3,
                                            kernelSize: 3,
                                            stride: 2,
                                            dataFormat: dataFormat)]
        for _ in 1 ..< resUnitPerBlock {
            self.blocks += [PreactResidualBlock(featureIn: depth3,
                                                featureOut: depth3,
                                                kernelSize: 3,
                                                stride: 1,
                                                dataFormat: dataFormat)]
        }
        self.blocks += [PreactResidualBlock(featureIn: depth3,
                                            featureOut: depth4,
                                            kernelSize: 3,
                                            stride: 2,
                                            dataFormat: dataFormat)]
        for _ in 1 ..< resUnitPerBlock {
            self.blocks += [PreactResidualBlock(featureIn: depth4,
                                                featureOut: depth4,
                                                kernelSize: 3,
                                                stride: 1,
                                                dataFormat: dataFormat)]
        }
        self.dense1 = WeightNormDense(weight: Tensor(orthogonal: [depth4, 10]),
                                      bias: Tensor(zeros: [1,1]),
                                      g: Tensor(zeros: [10]))
        
        //self.projectUnitNorm()
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>{
        var tmp = conv1(input) * multiplier1 + bias1
        tmp = blocks.differentiableReduce(tmp) {last, layer in layer(last)}
        tmp = mish(tmp * multiplier2 + bias2)
        let squeezingAxes = dataFormat == .nchw ? [2, 3] : [1, 2]
        tmp = tmp.mean(squeezingAxes: squeezingAxes)
        tmp = dense1(tmp)
        //print(tmp.standardDeviation())
        return tmp
    }
    
    public mutating func replaceParameters(_ newValue: TangentVector) {
        multiplier1 = newValue.multiplier1
        conv1.replaceParameters(newValue.conv1)
        bias1 = newValue.bias1
        for ii in 0..<blocks.count {
            blocks[ii].replaceParameters(newValue.blocks[ii])
        }
        multiplier2 = newValue.multiplier2
        bias2 = newValue.bias2
        dense1.replaceParameters(newValue.dense1)
    }
    // No longer used as filters and weights use didSet to maintain normalized weights.
    public mutating func projectUnitNorm() {
        conv1.filter = conv1.filter.weightNormalized()
        for ii in 0 ..< blocks.count {
            //blocks[ii].shortcut.filter = blocks[ii].shortcut.filter.weightNormalized()
            blocks[ii].conv1.filter = blocks[ii].conv1.filter.weightNormalized()
            blocks[ii].conv2.filter = blocks[ii].conv2.filter.weightNormalized()
        }
        dense1.weight = dense1.weight.weightNormalized()
    }
}
