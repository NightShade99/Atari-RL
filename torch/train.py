
import os
import wandb
import torch
import pickle
import argparse 
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from utils import *
from networks import *
from data_utils import *
from datetime import datetime as dt


def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    states, actions = load_data(args.data_path)
    train_size = int(len(states) * args.train_split) 
    state_shape = states[0].shape
    
    train_states, train_actions = states[:train_size], actions[:train_size]
    test_states, test_actions = states[train_size:], actions[train_size:]
    train_dset = ExperienceDataset(train_states, train_actions)
    test_dset = ExperienceDataset(test_states, test_actions)
    
    # To preserve RAM, delete copy of states and actions
    del states, actions
    
    # Dataloaders
    trainloader = data.DataLoader(
        train_dset, 
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    testloader = data.DataLoader(
        test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers
    )
    
    # Model initialization
    model = ViTModel(
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_actions=args.num_actions,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        in_channels=state_shape[-1],
        seqlen=(state_shape[0] // args.patch_size) * (state_shape[1] // args.patch_size),
        mlp_hidden_dim=args.mlp_hidden_dim,
        attn_dropout_rate=args.attn_dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-10, last_epoch=-1)
    
    # Logging 
    expt_dir = os.path.join(args.out_dir, dt.now().strftime('%d-%m-%Y_%H-%M'))
    os.makedirs(expt_dir, exist_ok=True)
    logger = Logger(expt_dir)
    logger.record_args(args)
    
    if args.wandb:
        run = wandb.init(project='explaining-drl-with-self-attention')
        logger.write("Wandb run URL: {}".format(run.get_url()))
    
    def train_step(batch):
        states, actions = batch 
        states, actions = states.to(device), actions.to(device)
        
        output, _ = model(states)
        loss = F.cross_entropy(output, actions)
        acc = output.argmax(-1).eq(actions).float().mean().item()        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), acc
    
    @torch.no_grad()
    def eval_step(batch):
        states, actions = batch 
        states, actions = states.to(device), actions.to(device)

        output, _ = model(states)
        loss = F.cross_entropy(output, actions)
        acc = output.argmax(-1).eq(actions).float().mean().item()        
        return loss.item(), acc
    
    def evaluate(dataloader):
        meter = AvgMeter()
        model.eval()
        
        for step, batch in enumerate(dataloader):
            loss, acc = eval_step(batch)
            meter.add({'loss': loss, 'accuracy': acc})
            pbar(p=(step+1)/len(dataloader), msg=meter.msg())
            
        return meter.get()['loss'], meter.get()['accuracy']
    
    def save(epoch, model_state_dict, optim_state_dict):
        state = {'model': model_state_dict, 'optim': optim_state_dict, 'epoch': epoch}
        torch.save(state, os.path.join(expt_dir, 'ckpt.pth'))
            
    # Training 
    best_train_loss, best_train_acc = float('inf'), 0.0
    best_test_loss, best_test_acc = float('inf'), 0.0
    
    logger.print("Beginning training...", mode='info')
    for epoch in range(1, args.train_epochs+1):
        meter = AvgMeter()
        model.train()
        
        for step, batch in enumerate(trainloader):
            loss, acc = train_step(batch)
            meter.add({'loss': loss, 'accuracy': acc})
            pbar(p=(step+1)/len(trainloader), msg=meter.msg())
            break
            
        avg_train_loss, avg_train_acc = meter.get()['loss'], meter.get()['accuracy']
        avg_test_loss, avg_test_acc = evaluate(testloader)
        
        best_train_loss = min(best_train_loss, avg_train_loss)
        best_train_acc = max(best_train_acc, avg_train_acc)
        best_test_loss = min(best_test_loss, avg_test_loss)
        
        if avg_test_acc >= best_test_acc:
            save(epoch, model.state_dict(), optimizer.state_dict())
            best_test_acc = avg_test_acc
            
        scheduler.step()
            
        logger.record("[{}/{}] Train loss: {:.4f} - Train Acc: {:.4f} - Test loss: {:.4f} - Test Acc: {:.4f}".format(
            epoch, args.train_epochs, avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc
        ), mode='train')
        logger.record("[{}/{}] Best Train Acc: {:.4f} - Best Test Acc: {:.4f}".format(
            epoch, args.train_epochs, best_train_acc, best_test_acc
        ), mode='info')
        print("---------------------------------------------------------------------------------------------------")
        
        if args.wandb:
            wandb.log({
                'Train loss': avg_train_loss, 'Train accuracy': avg_train_acc,
                'Test loss': avg_test_loss, 'Test accuracy': avg_test_acc, 'Epoch': epoch
            })
        
        
if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--num_actions', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wandb', action='store_true', default=False)
    ap.add_argument('--out_dir', type=str, default='./out')
    ap.add_argument('--train_split', type=float, default=0.8)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--num_heads', type=int, default=1)
    ap.add_argument('--num_layers', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=4)
    ap.add_argument('--model_dim', type=int, default=512)
    ap.add_argument('--mlp_hidden_dim', type=int, default=2048)
    ap.add_argument('--attn_dropout_rate', type=float, default=0.1)
    ap.add_argument('--base_lr', type=float, default=0.0001)
    ap.add_argument('--weight_decay', type=int, default=1e-06)
    ap.add_argument('--train_epochs', type=int, default=500)
    args = ap.parse_args()
    
    main(args)