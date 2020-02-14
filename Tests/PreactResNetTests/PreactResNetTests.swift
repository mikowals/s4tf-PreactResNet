import XCTest
import TensorFlow

@testable import PreactResNet

final class PreactResNetTests: XCTestCase {
    func testL2NormGrad() {
        @differentiable(wrt: x)
        func norm(_ x: Tensor<Float>, axes: [Int]) -> Tensor<Float> {
            return TensorFlow.sqrt(x.squared().sum(alongAxes: axes))
        }
        
        let x = Tensor<Float>(randomNormal: [9, 9, 32, 64])
        let grad = gradient(at: x) { $0.l2Norm(alongAxes: [0, 1, 2]).sum() }
        let targetGrad = gradient(at: x) { norm($0, axes: [0, 1, 2]).sum() }
        XCTAssertTrue(grad.isAlmostEqual(to: targetGrad, tolerance: 1e-6))
        
        let y = Tensor<Float>(randomNormal: [256, 10])
        let grad2 = gradient(at: y) {$0.l2Norm(alongAxes: [0]).sum() }
        let targetGrad2 = gradient(at: y) { norm($0, axes: [0]).sum() }
        XCTAssertTrue(grad2.isAlmostEqual(to: targetGrad2, tolerance: 1e-6))
    }
    
    func testGetSetParameters() {
        var model = PreactResNet<Float>(dataFormat: _Raw.DataFormat.nhwc, denseG: 0)
        var parameters = model.differentiableVectorView
        let originalParameters = parameters
        var count = 0
        for kp in parameters.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            count += parameters[keyPath: kp].shape.contiguousSize
            parameters[keyPath: kp] = Tensor<Float>(repeating: 0.02,
                                                    shape: parameters[keyPath: kp].shape)
        }
        XCTAssertEqual(count, 7352745)
        model.replaceParameters(parameters)
        parameters = model.differentiableVectorView
        for kp in parameters.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            XCTAssertNotEqual(parameters[keyPath: kp], originalParameters[keyPath: kp])
        }
    }
    
    func testWeightNormConv2D(){
        var conv = WeightNormConv2D<Float>(filter: Tensor(orthogonal: [3, 3, 32, 64]),
                                           g: Tensor(zeros: [64]))
        XCTAssertTrue(
            conv.filter.l2Norm(alongAxes: [0, 1, 2]).isAlmostEqual(to: Tensor(ones: [1, 1, 1, 64]),
                                                                   tolerance: 1e-6),
            "initial weights normalized"
        )
        var direction = conv.differentiableVectorView
        direction.filter = Tensor<Float>(repeating: 0.2, shape: [3, 3, 32, 64])
        conv.move(along: direction)
        XCTAssertTrue(
            conv.filter.l2Norm(alongAxes: [0, 1, 2]).isAlmostEqual(to: Tensor(ones: [1, 1, 1, 64]),
                                                                   tolerance: 1e-6),
            "updated weights normalized"
        )
    }
    
    func testWeightNormDense(){
        var dense = WeightNormDense<Float>(weight: Tensor(orthogonal: [64, 10]),
                                           bias: Tensor(zeros: [10]),
                                           g: Tensor(zeros: [10]))
        XCTAssertTrue(
            dense.weight.l2Norm(alongAxes: [0]).isAlmostEqual(to: Tensor(ones: [1, 10]),
                                                              tolerance: 1e-6),
            "initial weights normalized"
        )
        var direction = dense.differentiableVectorView
        direction.weight = Tensor<Float>(repeating: 0.2, shape: [64, 10])
        dense.move(along: direction)
        XCTAssertTrue(
            dense.weight.l2Norm(alongAxes: [0]).isAlmostEqual(to: Tensor(ones: [1, 10]),
                                                              tolerance: 1e-6),
            "updated weights normalized"
        )
    }
    
    func testWeightPreactConv2D(){
        var preact = PreactConv2D<Float>(filter: Tensor(orthogonal: [3, 3, 32, 64]),
                                         bias1: Tensor(zeros: [64]),
                                         bias2: Tensor(zeros: [64]),
                                         g: Tensor(zeros: [64]))
        XCTAssertTrue(
            preact.filter.l2Norm(alongAxes: [0, 1, 2]).isAlmostEqual(to: Tensor(ones: [1, 1, 1, 64]),
                                                                     tolerance: 1e-6),
            "initial weights normalized"
        )
        var direction = preact.differentiableVectorView
        direction.filter = Tensor<Float>(repeating: 0.2, shape: [3, 3, 32, 64])
        preact.move(along: direction)
        XCTAssertTrue(
            preact.filter.l2Norm(alongAxes: [0, 1, 2]).isAlmostEqual(to: Tensor(ones: [1, 1, 1, 64]),
                                                                     tolerance: 1e-6),
            "updated weights normalized"
        )
    }

    static var allTests = [
        ("testL2NormGrad", testL2NormGrad),
        ("testGetSetParameters", testGetSetParameters),
        ("testWeightNormConv2D", testWeightNormConv2D),
        ("testWeightPreactConv2D", testWeightPreactConv2D),
        ("testWeightNormDense", testWeightNormDense),
    ]
}
