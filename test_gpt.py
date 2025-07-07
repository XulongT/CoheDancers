import torch
import torch.optim as optim
from utils.metric import TestMetric as Metric
from utils.savefig import savefig, save_model
from utils.load_model import load_model
from utils.visual import visual
from utils.utils import seed_everything
from models.GPT.dataloader import createEvalDataset
from models.GPT.trainer import Trainer
from datetime import datetime
from tqdm import tqdm
from utils.processor import Processor

def validate(model, device, eval_loader, epoch, root_dir, exp_name):
    model.eval()
    losses = []
    metric = Metric(root_dir)
    with torch.no_grad():
        for _, data in tqdm(enumerate(eval_loader)):
            result, loss = model.test(data, device)
            losses.append(loss['total'])
            metric.update(result)
            savefig(result, epoch, exp_name)
        save_model(model, epoch, exp_name)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Eval Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')
    metric.result()
    # visual(exp_name, epoch)

def main():

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    root_dir = './I-Dancers'
    exp_name = 'group_gpt'
    epoch = 500

    print('Loading Data')
    print('Construct Eval Data')
    eval_loader = createEvalDataset(root_dir=root_dir, batch_size=batch_size)
    print('Start Training!')

    processor = Processor(device, 'test')
    model_args = {'device': device, 'processor': processor}
    model = Trainer(**model_args).to(device)
    model = load_model(model, exp_name, epoch)
    
    validate(model, device, eval_loader, epoch, root_dir, exp_name)

if __name__ == '__main__':
    main()
