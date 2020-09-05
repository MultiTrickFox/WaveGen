import config
from ext import now
from model import make_model, respond_to
from model import load_model, save_model
from model import TorchModel
from model import sgd, adaptive_sgd
from data import load_data, split_dataset, batchify

from matplotlib.pyplot import plot, show


def main():

    print(f'readying model & data @ {now()}')

    data = load_data()

    model = load_model()
    if not model:
        model = make_model()

    print(f'total files: {len(data)}, ', end='')

    data, data_dev = split_dataset(data)

    if not config.batch_size:
        config.batch_size = len(data_dev)
    elif config.batch_size > len(data):
        config.batch_size = len(data)

    print(f'train: {len(data)}, dev: {len(data_dev)}, batch size: {config.batch_size}')

    print(f'hm train: {sum(len(datapoint) for datapoint in data)}, '
          f'hm dev: {sum(len(datapoint) for datapoint in data_dev)}, '
          f'learning rate: {config.learning_rate}, '
          f'optimizer: {config.optimizer}, '
          f'\ntraining for {config.hm_epochs} epochs.. ', end='\n')

    data_losss, dev_losss = [], [None]

    # print(f'initializing losses @ {now()}', flush=True)
    # data_losss.append(dev_loss(model, data))
    # dev_losss.append(dev_loss(model, data_dev))
    # print(f'initial losses: {data_losss, dev_losss}')

    print(f'training started @ {now()}', flush=True)

    for ep in range(config.hm_epochs):

        loss = 0

        for i, batch in enumerate(batchify(data)):

            batch_size = sum(len(sequence) for sequence in batch)

            loss += respond_to(model, batch)

            sgd(model, config.learning_rate, batch_size) if config.optimizer == 'sgd' else \
                adaptive_sgd(model, ep, config.learning_rate, batch_size)

        loss /= sum(len(sequence) for sequence in data)

        data_losss.append(loss)
        # dev_losss.append(dev_loss(model, data_dev))

        print(f'epoch {ep}, loss {loss}, dev loss {dev_losss[-1]}, completed @ {now()}', flush=True)

    # data_losss.append(dev_loss(model, data))
    # dev_losss.append(dev_loss(model, data_dev))

    print(f'training ended @ {now()} \nfinal losses: {[data_losss[-1], dev_losss[-1]]}', flush=True)
    show(plot(data_losss))
    show(plot(dev_losss))

    if input(f'Save model as {config.model_path}? (y/n): ').lower() == 'y':
        save_model(load_model(), config.model_path + '_prev')
        save_model(model)

    return model, [data_losss, dev_losss]





if __name__ == '__main__':
    main()