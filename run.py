from simclr import SimCLR
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



if __name__ == "__main__":
    main()
