from cei220.seeds import derive_seed, full_namespace


def test_seed_derivation_is_deterministic():
    namespace = full_namespace("cifar10", "smallcnn", "cei-namespace-v1", "model/init")
    first = derive_seed(20236573, namespace, "cei-v1")
    second = derive_seed(20236573, namespace, "cei-v1")
    assert first == second


def test_namespaces_and_combinations_are_separated():
    values = {
        derive_seed(20236573, full_namespace(dataset, model, "cei-namespace-v1", component), "cei-v1")
        for dataset in ("cifar10", "fashion_mnist")
        for model in ("smallcnn", "resnet18_small")
        for component in ("model/init", "data/loader/train", "augmentation/train")
    }
    assert len(values) == 12
