import TensorFlow
import LayersDataFormat

@differentiable
func mish<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    return input * tanh(softplus(input))
}

@differentiable
public func noise<Scalar: TensorFlowFloatingPoint>(_ input: Tensor<Scalar>,
                                                   standardDeviation: Scalar = 0.1
) -> Tensor<Scalar> {
    switch Context.local.learningPhase {
    case .training:
        let rnd = Tensor<Scalar>(randomNormal: input.shape,
                                 mean: Tensor(0),
                                 standardDeviation: Tensor(standardDeviation))
        return rnd + input
    case .inference:
        return input
    }
}

public extension Tensor where Scalar: TensorFlowFloatingPoint {
    init(channelwiseZeroMean shape: TensorShape){
        self.init(randomUniform: shape, lowerBound: Tensor<Scalar>(-1), upperBound: Tensor<Scalar>(1))
        self = self - self.mean(alongAxes: [0, 1])
        self = self / self.l2Norm(alongAxes: [0, 1, 2])
    }
    
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

public func makeStrides(stride: Int, dataFormat: _Raw.DataFormat) -> (Int, Int, Int, Int) {
    let strides: (Int, Int, Int, Int)
    switch dataFormat {
    case .nchw:
        strides = (1, 1, stride, stride)
    case .nhwc:
        strides = (1, stride, stride, 1)
    }
    return strides
}

public struct ReparameterizedConv2D: Layer {
    public var filter, g: Tensor<Float>
    @noDerivative var stride: Int = 1
    @noDerivative var dataFormat: _Raw.DataFormat = .nhwc
    @noDerivative var zmg: Float
    
    public init(filterShape: TensorShape,
         initialG: Float = 0.8,
         stride: Int = 1,
         dataFormat: _Raw.DataFormat = .nhwc,
         zmg: Float = 0.85) {
        self.filter = Tensor<Float>(channelwiseZeroMean: filterShape)
        self.g = Tensor<Float>(repeating: TensorFlow.log(initialG), shape: [filterShape[3]])
        self.stride = stride
        self.dataFormat = dataFormat
        self.zmg = zmg
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.convolved2DDF(
            withFilter: filter.withDerivative({ [zmg] (v: inout Tensor<Float>) in
                v -= v.mean(alongAxes: [0, 1]) * zmg }) * TensorFlow.exp(g),
            strides: makeStrides(stride: stride, dataFormat: dataFormat),
            padding: .same,
            dataFormat: dataFormat)
    }
}

public struct WeightNormConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var filter: Tensor<Scalar> {
        didSet { filter = filter.weightNormalized() }
    }

    var g: Tensor<Scalar>
    @noDerivative var stride: Int = 1
    @noDerivative var dataFormat: _Raw.DataFormat = .nhwc

    init(filter: Tensor<Scalar>,
         g: Tensor<Scalar>,
         stride: Int = 1,
         dataFormat: _Raw.DataFormat = .nhwc)
    {
        self.filter = filter
        self.g = TensorFlow.log(g)
        self.stride = stride
        self.dataFormat = dataFormat
        defer {
            self.filter = self.filter
        }
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>{
        return input.convolved2DDF(withFilter: filter * TensorFlow.exp(g),
                                   strides: makeStrides(stride: stride, dataFormat: dataFormat),
                                   padding: .same,
                                   dataFormat: dataFormat)
    }
    
    mutating func replaceParameters(_ newValue: TangentVector) {
        filter = newValue.filter
        g = newValue.g
    }
}

public struct WeightNormDense<Scalar: TensorFlowFloatingPoint>: Layer {
    public var weight: Tensor<Scalar> {
        didSet { weight = weight.weightNormalized() }
    }
    var bias, g: Tensor<Scalar>
    
    init(weight: Tensor<Scalar>, bias: Tensor<Scalar>, g: Tensor<Scalar>) {
        self.weight = weight
        self.bias = bias
        self.g = TensorFlow.log(g)
        defer {
            self.weight = self.weight
        }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return matmul(input + bias, weight * TensorFlow.exp(g)) //weight.weightNormalized(g: g))
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
    @noDerivative var dataFormat: _Raw.DataFormat = .nhwc
    @noDerivative let activation: Activation
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    init(filter: Tensor<Scalar>,
         bias1: Tensor<Scalar>,
         bias2: Tensor<Scalar>,
         g: Tensor<Scalar>,
         stride: Int = 1,
         activation: @escaping Activation = relu,
         dataFormat: _Raw.DataFormat = .nhwc)
    {
        self.filter = filter
        self.bias1 = bias1
        self.bias2 = bias2
        self.g = g
        self.stride = stride
        self.activation = activation
        self.dataFormat = dataFormat
        defer {
            self.filter = filter
        }
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let tmp = activation(input + bias1) + bias2
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

public struct Shortcut<Scalar: TensorFlowFloatingPoint>: Differentiable {
    @noDerivative let shortcutOp: @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    
    public init(stride: Int = 1, featureIncrease: Int = 0, dataFormat: _Raw.DataFormat = .nhwc) {
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
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return shortcutOp(input)
    }
}

public struct PreactResidualBlock<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative let stride: Int
    @noDerivative let featureIn: Int
    @noDerivative let featureOut: Int
    @noDerivative let shortcut: Shortcut<Scalar>
    var conv1: PreactConv2D<Scalar>
    var conv2: PreactConv2D<Scalar>
    var multiplier = Tensor<Scalar>(ones: [1,1,1,1])
    var bias = Tensor<Scalar>(zeros: [1,1,1,1])
    
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    init(
        featureIn: Int,
        featureOut: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        activation: @escaping Activation = relu,
        dataFormat: _Raw.DataFormat = .nhwc
    ) {
        self.stride = stride
        self.featureIn = featureIn
        self.featureOut = featureOut
        self.shortcut = Shortcut(stride: stride,
                                 featureIncrease: featureOut - featureIn,
                                 dataFormat: dataFormat)
        self.conv1 = PreactConv2D(
            filter: Tensor(channelwiseZeroMean: [kernelSize, kernelSize, featureIn, featureOut]),
            bias1: Tensor(zeros: [1,1,1,1]),
            bias2: Tensor(zeros: [1,1,1,1]),
            g: Tensor(ones: [featureOut]) * sqrt(Scalar(2 * featureIn) / Scalar(featureOut)),
            stride: stride,
            activation: activation,
            dataFormat: dataFormat
        )
        self.conv2 = PreactConv2D(
            filter: Tensor(channelwiseZeroMean: [kernelSize, kernelSize, featureOut, featureOut]),
            bias1: Tensor(zeros: [1,1,1,1]),
            bias2: Tensor(zeros: [1,1,1,1]),
            g: Tensor(ones: [featureOut]) * rsqrt(Tensor<Scalar>(5)),
            stride: 1,
            activation: activation,
            dataFormat: dataFormat
        )
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let tmp = conv2(conv1(noise(input)))
        let sc = shortcut(input)
        return tmp * multiplier + bias + sc
    }
    
    public mutating func replaceParameters(_ newValue: TangentVector) {
        conv1.replaceParameters(newValue.conv1)
        conv2.replaceParameters(newValue.conv2)
        multiplier = newValue.multiplier
        bias = newValue.bias
    }
}

public struct PreactResNet<Scalar: TensorFlowFloatingPoint>: Layer {
    @noDerivative var dataFormat: _Raw.DataFormat = .nhwc
    @noDerivative let activation: Activation
    @noDerivative let denseG: Float
    var multiplier1 = Tensor<Scalar>(ones: [1,1,1,1])
    var bias1 = Tensor<Scalar>(zeros: [1,1,1,1])
    var conv1: WeightNormConv2D<Scalar>
    var blocks: [PreactResidualBlock<Scalar>] = []
    var multiplier2 = Tensor<Scalar>(ones: [1,1,1,1])
    var bias2 = Tensor<Scalar>(zeros: [1,1,1,1])
    var dense1: WeightNormDense<Scalar>
    
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

    public init(
        activation: @escaping Activation = relu,
        dataFormat: _Raw.DataFormat = .nhwc,
        denseG: Float = 0
    ) {
        self.activation = activation
        self.dataFormat = dataFormat
        self.denseG = denseG
        let depth = 16
        let depth2 = 64
        let depth3 = 128
        let depth4 = 256
        let resUnitPerBlock = 5
        let blockSpecs = [
            (depthIn: depth, depthOut: depth2, stride: 1),
            (depthIn: depth2, depthOut: depth3, stride: 2),
            (depthIn: depth3, depthOut: depth4, stride: 2)
        ]
        
        self.conv1 = WeightNormConv2D(
            filter: Tensor(channelwiseZeroMean: [3, 3, 3, depth]),
            g: Tensor(ones: [depth]) * sqrt(Scalar(2 * 3) / Scalar(depth)),
            stride: 1,
            dataFormat: dataFormat
        )
        
        for s in blockSpecs {
            self.blocks += [PreactResidualBlock(featureIn: s.depthIn,
                                                featureOut: s.depthOut,
                                                kernelSize: 3,
                                                stride: s.stride,
                                                activation: activation,
                                                dataFormat: dataFormat)]
            for _ in 1 ..< resUnitPerBlock {
                self.blocks += [PreactResidualBlock(featureIn: s.depthOut,
                                                    featureOut: s.depthOut,
                                                    kernelSize: 3,
                                                    stride: 1,
                                                    activation: activation,
                                                    dataFormat: dataFormat)]
            }
            
        }
        
        self.dense1 = WeightNormDense(weight: Tensor(orthogonal: [depth4, 10]),
                                      bias: Tensor(zeros: [1,1]),
                                      g: Tensor(repeating: Scalar(denseG), shape: [10]))
        
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>{
        var tmp = conv1(noise(input)) * multiplier1 + bias1
        tmp = blocks.differentiableReduce(tmp) {last, layer in layer(last)}
        tmp = activation(tmp * multiplier2 + bias2)
        let squeezingAxes = dataFormat == .nchw ? [2, 3] : [1, 2]
        tmp = tmp.mean(squeezingAxes: squeezingAxes)
        return dense1(tmp)
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

