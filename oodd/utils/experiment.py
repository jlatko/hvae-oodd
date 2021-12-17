# Define dataset groups

BNW_BINARY = [
    "FashionMNISTBinarized",
    "MNISTBinarized",
    "notMNISTBinarized",
    "Omniglot28x28Binarized",
    "Omniglot28x28InvertedBinarized",
    "KMNISTBinarized",
]
BNW_CONT = [
    "FashionMNISTDequantized",
    "MNISTDequantized",
    "notMNISTDequantized",
    "Omniglot28x28Dequantized",
    "Omniglot28x28InvertedDequantized",
    "SmallNORB28x28Dequantized",
]

COLOR = [
    "CIFAR10Dequantized",
    "CIFAR100Dequantized",
    "SVHNDequantized",
]

CIFAR_TOPICS = [
    "CIFAR10DequantizedAnimals",
    "CIFAR10DequantizedTransportation",
    "CIFAR100DequantizedAnimals",
    "CIFAR100DequantizedPlants",
    "CIFAR100DequantizedThings",
    "CIFAR100DequantizedRest",
]

CIFAR_ODD_EVEN = [
    "CIFAR10DequantizedOdd",
    "CIFAR10DequantizedEven",
    "CIFAR100DequantizedOdd",
    "CIFAR100DequantizedEven",
]

CIFAR_HALVES = [
    "CIFAR10DequantizedFirstHalf",
    "CIFAR10DequantizedSecondHalf",
    "CIFAR100DequantizedFirstHalf",
    "CIFAR100DequantizedSecondHalf",
]
