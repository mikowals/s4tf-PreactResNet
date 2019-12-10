// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PreactResNet",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "PreactResNet",
            targets: ["PreactResNet"]),
    ],
    dependencies: [
        .package(url: "https://github.com/mikowals/s4tf-LayersDataFormat.git", .revision("c1a5ee9d2aef8551d278084b50d62a666ed73149")),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "PreactResNet",
            dependencies: ["LayersDataFormat"]),
        .testTarget(
            name: "PreactResNetTests",
            dependencies: ["PreactResNet"]),
    ]
)
