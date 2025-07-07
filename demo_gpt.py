import torch
import torch.optim as optim
from utils.savefig import savefig, save_model
from utils.load_model import load_model
from utils.visual import visual
from utils.utils import seed_everything
from models.GPT.dataloader import createDemoDataset
from models.GPT.trainer import Trainer
from datetime import datetime
from tqdm import tqdm
from utils.processor import Processor
import time

def validate(model, device, demo_loader, epoch, root_dir, exp_name):
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(demo_loader)):
            start_time = time.time()
            model.demo(data, device)
            end_time = time.time()
            dance_duration = (data['music_librosa'].shape[1]) / 30.0
            print(f"Model Prediction Duration: {end_time-start_time:.2f}s")
            print(f"Dance Duration: {dance_duration:.2f}s")

def main():

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    root_dir = './demo'
    exp_name = 'group_gpt'
    epoch = 500

    print('Loading Data')
    print('Construct Demo Data')
    demo_loader = createDemoDataset(root_dir=root_dir, batch_size=batch_size)
    print('Start Training!')

    processor = Processor(device, 'test')
    model_args = {'device': device, 'processor': processor}
    model = Trainer(**model_args).to(device)
    model = load_model(model, exp_name, epoch)
    
    validate(model, device, demo_loader, epoch, root_dir, exp_name)

if __name__ == '__main__':
    main()
