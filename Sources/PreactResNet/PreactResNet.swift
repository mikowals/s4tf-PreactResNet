import TensorFlow
import LayersDataFormat

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
        return centered / centered.l2Norm(alongAxes: axes)
    }
}

struct WeightNormConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    var filter, g: Tensor<Scalar>
    @noDerivative let stride: Int
    /*var differentiableVectorView: TangentVector {
        get { TangentVector(filter: filter, g: g) }
        set { filter = newValue.filter; g = newValue.g }
    }*/

    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>{
        return input.convolved2DDF(withFilter: filter * g, //filter.weightNormalized(g: g),
                                    strides: (1, 1, stride, stride),
                                    padding: .same, dataFormat: .nchw)
    }
    
    mutating func replaceParameters(_ newValue: TangentVector) {
        filter = newValue.filter
        g = newValue.g
    }
}

struct WeightNormDense<Scalar: TensorFlowFloatingPoint>: Layer {
    var weight, bias, g: Tensor<Scalar>
    /*var differentiableVectorView: TangentVector {
        get { TangentVector(weight: weight, bias: bias, g: g) }
        set { weight = newValue.weight; bias = newValue.bias; g = newValue.g }
    }*/

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
    var filter, bias1, bias2, g: Tensor<Scalar>
    @noDerivative let stride: Int
    /*var differentiableVectorView: TangentVector {
        get { TangentVector(filter: filter, bias1: bias1, bias2: bias2, g: g) }
        set {
            filter = newValue.filter
            bias1 = newValue.bias1
            bias2 = newValue.bias2
            g = newValue.g
        }
    }*/

    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        let tmp = relu(input + bias1) + bias2
        return tmp.convolved2DDF(withFilter: filter * g, //filter.weightNormalized(g: g),
                                strides: (1, 1, stride, stride),
                                padding: .same, dataFormat: Raw.DataFormat.nchw)
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
    
    init(stride: Int = 1, featureIncrease: Int = 0){
        if stride > 1 || featureIncrease != 0 {
            self.shortcutOp = {input in
                          let tmp = input.averagePooledDF(kernelSize: (1,1,stride,stride),
                                                   strides: (1,1,stride,stride),
                                                   padding: .same,
                                                   dataFormat: .nchw)
                          return tmp.padded(forSizes: [(before: 0, after: 0),
                                                       (before: 0, after: featureIncrease),
                                                       (before: 0, after: 0),
                                                       (before: 0, after: 0),])
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
    @noDerivative var dataFormat: Raw.DataFormat = .nchw
    @noDerivative let isExpansion: Bool
    //var shortcut: WeightNormConv2D<Scalar>
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
        stride: Int = 1
    ) {
        let padding = kernelSize == 3 ? Padding.same : Padding.valid
        self.stride = stride
        self.featureIn = featureIn
        self.featureOut = featureOut
        //let shortcutType = stride == 2 ? ShortcutType.pooling :  ShortcutType.id
        /*
        self.shortcut = WeightNormConv2D(filter: Tensor(orthogonal: [1,1,featureIn, featureOut]),
                                         g: Tensor(ones: [featureOut]) * sqrt(Scalar(featureIn) / Scalar(featureOut)),
                                         stride: stride)
        */
        self.shortcut = Shortcut(stride: stride,
                            featureIncrease: featureOut - featureIn)
        self.isExpansion = featureIn != featureOut || stride != 1
        self.conv1 = PreactConv2D(
            filter: Tensor(orthogonal: [kernelSize, kernelSize, featureIn, featureOut]),
            bias1: Tensor(zeros: [1,1,1,1]),
            bias2: Tensor(zeros: [1,1,1,1]),
            g: Tensor(ones: [featureOut]) * sqrt(Scalar(2 * featureIn) / Scalar(featureOut)),
            stride: stride)
        self.conv2 = PreactConv2D(
            filter: Tensor(orthogonal: [kernelSize, kernelSize, featureOut, featureOut]),
            bias1: Tensor(zeros: [1,1,1,1]),
            bias2: Tensor(zeros: [1,1,1,1]),
            g: Tensor(ones: [featureOut]) * rsqrt(Tensor<Scalar>(5)),
            stride: 1)
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        //print("in: ", input.standardDeviation())
        let tmp = conv2(conv1(input))
        /*let sc: Tensor<Scalar>
        if isExpansion {
            sc = shortcut(input)
        } else {
            sc = input
        }*/
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

    public init() {
        let depth = 16
        let depth2 = 64
        let depth3 = 128
        let depth4 = 256
        let resUnitPerBlock = 5

        self.conv1 = WeightNormConv2D(
            filter: Tensor(orthogonal: [3, 3, 3, depth]),
            g: Tensor(ones: [depth]) * sqrt(Scalar(2 * 3) / Scalar(depth)),
            stride: 1)
        
        self.blocks = [PreactResidualBlock(featureIn: depth, featureOut: depth2, kernelSize: 3, stride: 1)]
        for _ in 1 ..< resUnitPerBlock {
            self.blocks += [PreactResidualBlock(featureIn: depth2, featureOut: depth2, kernelSize: 3, stride: 1)]
        }
        self.blocks += [PreactResidualBlock(featureIn: depth2, featureOut: depth3, kernelSize: 3, stride: 2)]
        for _ in 1 ..< resUnitPerBlock {
            self.blocks += [PreactResidualBlock(featureIn: depth3, featureOut: depth3, kernelSize: 3, stride: 1)]
        }
        self.blocks += [PreactResidualBlock(featureIn: depth3, featureOut: depth4, kernelSize: 3, stride: 2)]
        for _ in 1 ..< resUnitPerBlock {
            self.blocks += [PreactResidualBlock(featureIn: depth4, featureOut: depth4, kernelSize: 3, stride: 1)]
        }
        self.dense1 = WeightNormDense(weight: Tensor(orthogonal: [depth4, 10]),
                                      bias: Tensor(zeros: [1,1]),
                                      g: Tensor(zeros: [10])) //* sqrt(Scalar(depth4) / Scalar(10)))
        
        self.projectUnitNorm()
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar>{
        var tmp = conv1(input) * multiplier1 + bias1
        tmp = blocks.differentiableReduce(tmp) {last, layer in layer(last)}
        tmp = relu(tmp * multiplier2 + bias2)
        tmp = tmp.mean(squeezingAxes: [2,3])
        tmp = dense1(tmp)
        //print(tmp.standardDeviation())
        return tmp
    }
    
    mutating func replaceParameters(_ newValue: TangentVector) {
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
    mutating func projectUnitNorm() {
        conv1.filter = conv1.filter.weightNormalized()
        for ii in 0 ..< blocks.count {
            //blocks[ii].shortcut.filter = blocks[ii].shortcut.filter.weightNormalized()
            blocks[ii].conv1.filter = blocks[ii].conv1.filter.weightNormalized()
            blocks[ii].conv2.filter = blocks[ii].conv2.filter.weightNormalized()
        }
        dense1.weight = dense1.weight.weightNormalized()
    }
}
