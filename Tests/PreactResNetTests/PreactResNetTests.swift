import XCTest
import TensorFlow

@testable import PreactResNet

final class PreactResNetTests: XCTestCase {
    func testGetSetParameters() {
        var model = PreactResNet<Float>()
        var parameters = model.differentiableVectorView
        var count = 0
        for kp in parameters.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            count += parameters[keyPath: kp].shape.contiguousSize
            parameters[keyPath: kp] = Tensor<Float>(repeating: 2, shape: parameters[keyPath: kp].shape)
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
    }

    static var allTests = [
        ("testGetSetParameters", testGetSetParameters),
    ]
}
