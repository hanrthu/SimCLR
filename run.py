from simclr import SimCLR
from logistic_evaluator import ResNetFeatureExtractor,LogiticRegressionEvaluator
import yaml
from data_aug.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    simclr = SimCLR(dataset, config)
    if config['mode'] == 'train':
        simclr.train()
    if config['mode'] == 'test':
        simclr.test()
    if config['mode'] == 'eval':
        logistic = ResNetFeatureExtractor(config)
        X_train_feature, y_train, X_test_feature, y_test = logistic.get_resnet_features()
        classifier = LogiticRegressionEvaluator(2048,10)
        classifier.train(X_train_feature, y_train, X_test_feature, y_test)



if __name__ == "__main__":
    main()
