import XCTest
import TensorFlow

@testable import PreactResNet

final class PreactResNetTests: XCTestCase {
    /*func testGetSetParameters() {
        var model = PreactResNet<Float>(dataFormat: Raw.DataFormat.nhwc)
        var parameters = model.differentiableVectorView
        var count = 0
        for kp in parameters.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            count += parameters[keyPath: kp].shape.contiguousSize
            parameters[keyPath: kp] = Tensor<Float>(repeating: 0.02, shape: parameters[keyPath: kp].shape)
        }
        model.replaceParameters(parameters)
        parameters = model.differentiableVectorView
        var sum = Tensor<Float>(0)
        for kp in parameters.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            sum += parameters[keyPath: kp].sum()
        }
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(count, 7352745)
        XCTAssertEqual(sum, Tensor<Float>(7352745 * 2))
    }*/
    
    func testWeightNormalization(){
        var conv = WeightNormConv2D<Float>(filter: Tensor(orthogonal: [3, 3, 2, 1]),
                                          g: Tensor(zeros: [2]))
        print ("norm: ", conv.filter.l2Norm(alongAxes: [0,1,2]))
        XCTAssertTrue((conv.filter.l2Norm(alongAxes: [0, 1, 2]) .== 1).all(),
                      "initial weights normalized")
        var direction = conv.differentiableVectorView
        direction.filter = Tensor<Float>(repeating: 0.2, shape: [3, 3, 2, 1])
        conv.move(along: direction)
        XCTAssertTrue((conv.filter.l2Norm(alongAxes: [0, 1, 2]) .== 1).all(),
                      "updated weights normalized")
    }

    static var allTests = [
        ("testWeightNormalization", testWeightNormalization),
    ]
}
