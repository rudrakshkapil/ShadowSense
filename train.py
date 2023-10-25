import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from dataloaders import RGBxThermalTreeCrownDataset
from archs.RGBTdetector import RGBxThermalDetector
from archs.encDec import UNet, Discriminator

from torch.autograd import Variable

from df_repo.deepforest import main as df_main

def train(save_dir):
    # data loaders
    train_dataset = RGBxThermalTreeCrownDataset(root='data', split='train', masks_dir='masks')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    # for logging
    writer = SummaryWriter(save_dir)
    checkpoint_dir = f'{save_dir}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # model
    weights_path = 'df_retinanet.pt'
    model = RGBxThermalDetector(weights_path)
    model.train()
    model.cuda()
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = ReduceLROnPlateau(optimizer, patience=2)
    scheduler = ExponentialLR(optimizer, gamma=0.9)


    # start = 2000
    # save_path = f"output/tm=whole/da=single_theirs/fm=bg_avg - burnin+training(best) - alternate freezing/checkpoints/chkpt_2000.pt"
    # chkpt = torch.load(save_path)
    # model.load_state_dict(chkpt['model_state_dict'])
    # optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    # scheduler.load_state_dict(chkpt)
        # 'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        # },

    # loop
    n = 1
    current_epoch = 0
    max_iter = 10000
    save_step = 1000
    data_len = len(train_dataloader)
    num_epochs = max_iter / data_len
    loader_iter = iter(train_dataloader)
    loss = None
    for i in (pbar:=tqdm(range(max_iter))):
        pbar.set_description(f"{loss}")
        # get current batch (reset between epochs)
        try:
            current_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_dataloader)
            current_batch = next(loader_iter)
            current_epoch += 1
            n = 1
            scheduler.step()

        # at start only train discriminators (burn in stage)
        if i == 0:
            for name, param in model.thermal_prelayer.named_parameters():
                param.requires_grad = False
            for name, param in model.thermal_detector.named_parameters():
                param.requires_grad = False
            alphas = [0.0, 0.0, 0.0]
        # else unfreeze (at this point, discriminator should be good, train thermal backbone as well, with full gradient reversal)
        elif i == burn_in_iters:
            for name, param in model.thermal_prelayer.named_parameters():
                param.requires_grad = True
            for name, param in model.thermal_detector.named_parameters():
                if name.startswith("backbone.body"): # NOTE: only resnet unfrozen
                    param.requires_grad = True
            alphas = [1.0, 1.0, 1.0]

            for name, param in model.DA_res3.named_parameters():
                param.requires_grad = False
            for name, param in model.DA_res4.named_parameters():
                param.requires_grad = False
            for name, param in model.DA_res5.named_parameters():
                param.requires_grad = False

        #     # for name, param in model.DA_FPN3.named_parameters():
        #     #     param.requires_grad = False
        #     # for name, param in model.DA_FPN4.named_parameters():
        #     #     param.requires_grad = False
        #     # for name, param in model.DA_FPN5.named_parameters():
        #     #     param.requires_grad = False
            


        optimizer.zero_grad()

        # get inputs
        rgb_img, thm_img, masks = current_batch
        rgb_img, thm_img, masks = rgb_img.cuda(), thm_img.cuda(), masks.cuda()
        # rgb_img, thm_img = current_batch
        # rgb_img, thm_img = rgb_img.cuda(), thm_img.cuda()

        # alpha scaling for GRL (to overcome noisy discriminator noise at early training stages)
        # p = float(n + current_epoch*data_len) / num_epochs / data_len
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # alphas = [alpha, alpha, alpha]
        # alphas = [max(alpha, 0.5), max(alpha, 0.5), max(alpha, 0.3)]
        # alphas = [1.0, 1.0, 1.0]
        n+=1

        # send through model (returns losses_dict)
        # losses_dict = model(rgb_img, thm_img, alphas)
        losses_dict = model(rgb_img, thm_img, alphas, masks)
        for key,loss in losses_dict.items():
            writer.add_scalar(f"Loss/train/{key}", loss, i)

        # backprop
        if i < burn_in_iters:
            # losses_dict.pop('loss_distill_cls_logits')
            # losses_dict.pop('loss_distill_bbox_regression')
            losses_dict.pop('loss_fpn')
        else:
            losses_dict['loss_fpn'] *= 0.005

        loss = sum(losses_dict.values()) #** (-1.0) # NOTE: check if need *-1 here or ^-1 => no need, worse
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        # save 
        if i > 0 and i % save_step == 0:
            save_path = f"{checkpoint_dir}/chkpt_{i}.pt"
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)

    # save final model
    save_path = f"{checkpoint_dir}/final_{i+1}.pt"
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }, save_path)
    
def train_resume(save_dir):
    # data loaders
    train_dataset = RGBxThermalTreeCrownDataset(root='data', split='train', masks_dir='masks')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    # for logging
    writer = SummaryWriter(save_dir)
    checkpoint_dir = f'{save_dir}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # model
    weights_path = 'df_retinanet.pt'
    model = RGBxThermalDetector(weights_path)
    model.train()
    model.cuda()
    model.rgb_detector.eval()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


    # optimizer & scheduler
    # optimizer = torch.optim.Adam(model.parameters())

    # NOTE: turn on
    # optimizer_dis = torch.optim.Adam(params=list(model.DA_res3.parameters()) + list(model.DA_res4.parameters()) + list(model.DA_res5.parameters()))
    # optimizer_res = torch.optim.Adam(params=list(model.thermal_prelayer.parameters()) + list(model.thermal_detector.backbone.body.parameters()))
    
    # optimizer_fpn = torch.optim.Adam(params=list(model.thermal_prelayer.parameters()) + list(model.thermal_detector.backbone.parameters()))

    # NOTE: for 3 or 2 stages
    # optimizer_res = torch.optim.Adam(params=list(model.thermal_prelayer.parameters()) + list(model.thermal_detector.backbone.body.parameters()) + list(model.DA_res3.parameters()) + list(model.DA_res4.parameters()) + list(model.DA_res5.parameters()))
    # optimizer_fpn = torch.optim.Adam(params=list(model.thermal_detector.backbone.fpn.parameters()))

    # scheduler_res = ExponentialLR(optimizer_res, gamma=0.9)
    # scheduler_fpn = ExponentialLR(optimizer_fpn, gamma=0.9)

    # NOTE: 
    optimizer = torch.optim.Adam(params = model.parameters())
    scheduler = ExponentialLR(optimizer, gamma=0.9)


    start = 0
    # start = 13000
    # save_path = f"output/different optims, 1x (best!)/checkpoints/chkpt_13000.pt"
    # chkpt = torch.load(save_path)
    # model.load_state_dict(chkpt['model_state_dict'])
    # optimizer_res.load_state_dict(chkpt['optimizer_res_state_dict'])
    # # optimizer_fpn.load_state_dict(chkpt['optimizer_fpn_state_dict'])
    # scheduler_res.load_state_dict(chkpt['scheduler_res_state_dict'])
    # # scheduler_fpn.load_state_dict(chkpt['scheduler_fpn_state_dict'])

    # add fpn to optim
    # optimizer_fpn.param_groups.append({'params', model.thermal_detector.backbone.fpn.parameters()})

    # for groups in optimizer_fpn.param_groups: groups['lr']/=10 # NOTE: reduced LR
    

    # loop
    
    burn_in_iters = 3000# 3000 #3000
    n = 1
    current_epoch = 0
    max_iter = 10000# 17000 #15000
    save_step =  1000#1000 #1000
    fpn_iters = 8000# 13000
    data_len = len(train_dataloader)
    max_epoch = max_iter / data_len
    loader_iter = iter(train_dataloader)
    loss = None
    for i in (pbar:=tqdm(range(start,max_iter))):
        pbar.set_description(f"{loss}")
        # get current batch (reset between epochs)
        try:
            current_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_dataloader)
            current_batch = next(loader_iter)
            current_epoch += 1
            n = 1
            
            # NOTE: for 2/3 stages
            # scheduler_res.step() # NOTE turn on
            # if i >= fpn_iters:
            #     scheduler_fpn.step()

            scheduler.step()

        # at start only train discriminators (burn in stage)
        # if (i == start and start < burn_in_iters) or i == 0:
        #     print("Freezing Thermal Prelayer + Backbone\nDiscriminators Unfrozen")
        #     for param in model.thermal_prelayer.parameters():
        #         param.requires_grad = False
        #         param.grad = None
        #     for param in model.thermal_detector.parameters():
        #         param.requires_grad = False
        #         param.grad = None
        #     alphas = [0.0, 0.0, 0.0]

        # # else unfreeze (at this point, discriminator should be good, train thermal backbone as well, with full gradient reversal)
        # # elif == burn_in_iters:
        # if i == burn_in_iters or (i == start and start >= burn_in_iters) :
        #     print("Unfreezeing Thermal Prelayer + backbone\nFreezing Discriminators\n\n")
        #     for param in model.thermal_prelayer.parameters():
        #         param.requires_grad = True
        #         param.grad = torch.zeros_like(param)
        #     for param in model.thermal_detector.backbone.parameters(): # backbone => both body and fpn
        #         param.requires_grad = True
        #         param.grad = torch.zeros_like(param)
        #     alphas = [1.0, 1.0, 1.0]

        #     for param in model.DA_res3.parameters(): # NOTE: turn on
        #         param.requires_grad = False
        #         param.grad = None
        #     for param in model.DA_res4.parameters():
        #         param.requires_grad = False
        #         param.grad = None
        #     for param in model.DA_res5.parameters():
        #         param.requires_grad = False
        #         param.grad = None

        # if i == fpn_iters or (i == start and start >= fpn_iters) :
        #     print("Freezing Thermal Prelayer + Backbone\nUnfreezeing FPN")
        #     for param in model.thermal_prelayer.parameters():
        #         param.requires_grad = False
        #         param.grad = None
        #     for param in model.thermal_detector.backbone.body.parameters():
        #         param.requires_grad = False
        #         param.grad = None
        #     for param in model.thermal_detector.backbone.fpn.parameters():
        #         param.requires_grad = True
        #         param.grad = torch.zeros_like(param)


        #     # for name, param in model.DA_FPN3.named_parameters():
        #     #     param.requires_grad = False
        #     # for name, param in model.DA_FPN4.named_parameters():
        #     #     param.requires_grad = False
        #     # for name, param in model.DA_FPN5.named_parameters():
        #     #     param.requires_grad = False


        # NOTE: for 3 stages (alogn with 8 lines below)
        # if i == burn_in_iters or (i == start and start >= burn_in_iters):
        #     model.DA_res3.eval(); model.DA_res4.eval(); model.DA_res5.eval()

        # NOTE: for GRL then FPN (2 stages)
        # if i == fpn_iters or (i == start and start >= fpn_iters):
        #     model.DA_res3.eval(); model.DA_res4.eval(); model.DA_res5.eval()
        #     model.thermal_detector.backbone.body.eval()
            

        # get inputs
        rgb_img, thm_img, masks = current_batch
        rgb_img, thm_img, masks = rgb_img.cuda(), thm_img.cuda(), masks.cuda()
        n+=1

        # alphas
        # if i < burn_in_iters:
        #     alphas = [0.0, 0.0, 0.0]
        # else: #if i < fpn_iters:
        #     alphas = [1.0, 1.0, 1.0]

        p = float( n + current_epoch * data_len) / max_epoch / data_len
        alpha = 2. / ( 1. + np.exp( -10 * p)) - 1
        # alphas = [min(0.5,alpha), min(0.5,alpha), min(0.1,alpha)]
        alphas = [alpha, alpha, alpha]


        losses_dict = model(rgb_img, thm_img, alphas, masks, comp_fpn_loss = True)
        for key,loss in losses_dict.items():
            writer.add_scalar(f"Loss/train/{key}", loss, i)

        def plot_grad_flow(named_parameters):
            '''Plots the gradients flowing through different layers in the net during training.
            Can be used for checking for possible gradient vanishing / exploding problems.
            
            Usage: Plug this function in Trainer class after loss.backwards() as 
            "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
            ave_grads = []
            max_grads= []
            layers = []
            total,nonzero  = 1,1
            for n, p in named_parameters:
                total += 1
                if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
                    nonzero +=1
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
                    max_grads.append(p.grad.abs().max().detach().cpu().numpy())
            plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
            plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
            plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
            plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
            plt.xlim(left=0, right=len(ave_grads))
            plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow")
            plt.grid(True)
            plt.legend([Line2D([0], [0], color="c", lw=4),
                        Line2D([0], [0], color="b", lw=4),
                        Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
            plt.show()
            print(nonzero, layers, total)

        # resnet/disc loss
        disc_loss = sum([v for k,v in losses_dict.items() if 'fpn' not in k])
        loss_fpn = losses_dict['loss_fpn']
        
        # NOTE: for combined -- first one is correct, second is ablation (fpn loss only)
        loss = disc_loss + loss_fpn  # TODO: backward different parts loss_fpn.backward(inputs=model.thermal_detector.backbone.fpn.parameters())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        loss = f"[{np.round(disc_loss.item(),5)} + {np.round(loss_fpn.item(),5)} = {np.round(disc_loss.item(),5) +np.round(loss_fpn.item(),5)}]"

        # if i < burn_in_iters:
        #     optimizer_dis.zero_grad() # NOTE: turn on
        #     disc_loss.backward(inputs=list(model.DA_res3.parameters()) + list(model.DA_res4.parameters()) + list(model.DA_res5.parameters()))
        #     optimizer_dis.step()
        # if i < fpn_iters:
        # # elif i < fpn_iters:
        #     optimizer_res.zero_grad()
        #     # disc_loss.backward(inputs=list(model.thermal_prelayer.parameters()) + list(model.thermal_detector.backbone.body.parameters()))
        #     disc_loss.backward(inputs=list(model.thermal_prelayer.parameters()) + list(model.thermal_detector.backbone.body.parameters()) + list(model.DA_res3.parameters()) + list(model.DA_res4.parameters()) + list(model.DA_res5.parameters()))
        #     optimizer_res.step()
        # else:
        #     optimizer_fpn.zero_grad()
        #     loss_fpn.backward(inputs=list(model.thermal_detector.backbone.fpn.parameters()))
        #     optimizer_fpn.step()

        

        


        
        # # fpn loss
        # if i < fpn_iters:
        #     # losses_dict.pop('loss_distill_cls_logits')
        #     # losses_dict.pop('loss_distill_bbox_regression')
        #     # losses_dict.pop('loss_fpn')
        #     pass
        # else:
        #     # losses_dict['loss_fpn'] = 100/(losses_dict['loss_fpn'] + 1e-6)
        #     # losses_dict['loss_fpn'] *= -11 # TODO: try changing this
        #     # loss_fpn = losses_dict['loss_fpn']
        #     # writer.add_scalar(f"Loss/train/loss_fpn_before", losses_dict['loss_fpn'], i)
        #     loss_fpn = Variable(losses_dict['loss_fpn'], requires_grad = True)
        #     optimizer_fpn.zero_grad()
        #     loss_fpn.backward(inputs=list(model.thermal_detector.backbone.fpn.parameters())) # NOTE: turn on
        #     # writer.add_scalar(f"Loss/train/loss_fpn", loss_fpn, i)
        #     optimizer_fpn.step()
        #     # print(losses_dict['loss_fpn'])
        #     # l oss_fpn.backward()

        #     # NOTE: turn on
        #     loss = [np.round(loss.item(),5), np.round(loss_fpn.item(), 5), '=', np.round(loss.item()+loss_fpn.item(), 5)]

        # loss = sum(losses_dict.values()) #** (-1.0) # NOTE: check if need *-1 here or ^-1 => no need, worse
        # loss.backward()
        
        # optimizer.step()
        # scheduler.step(loss)

        # save 
        if i > 0 and i % save_step == 0:
            save_path = f"{checkpoint_dir}/chkpt_{i}.pt"
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'optimizer_res_state_dict': optimizer_res.state_dict(),
                # 'scheduler_res_state_dict': scheduler_res.state_dict(),  #NOTE: turn this and above on
                # 'optimizer_fpn_state_dict': optimizer_fpn.state_dict(),
                # 'scheduler_fpn_state_dict': scheduler_fpn.state_dict(),
                }, save_path)

    # save final model
    save_path = f"{checkpoint_dir}/chkpt_{i+1}.pt"
    torch.save({
            'epoch': i+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # 'optimizer_res_state_dict': optimizer_res.state_dict(),
            # 'scheduler_res_state_dict': scheduler_res.state_dict(), #NOTE: turn this and above on
            # 'optimizer_fpn_state_dict': optimizer_fpn.state_dict(),
            # 'scheduler_fpn_state_dict': scheduler_fpn.state_dict(),
            }, save_path)

# 
def train_enc_dec(save_dir):
    # data loaders
    train_dataset = RGBxThermalTreeCrownDataset(root='data', split='train', masks_dir='masks')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    # for logging
    writer = SummaryWriter(save_dir)
    checkpoint_dir = f'{save_dir}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # model
    model = UNet(1, 3)
    model.cuda()

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, patience=500)

    # loop
    n = 1
    current_epoch = 0
    max_iter = 10000
    save_step = 1000
    data_len = len(train_dataloader)
    num_epochs = max_iter / data_len
    loader_iter = iter(train_dataloader)
    loss = None
    for i in (pbar:=tqdm(range(max_iter))):
        pbar.set_description(f"{loss}")
        # get current batch (reset between epochs)
        try:
            current_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_dataloader)
            current_batch = next(loader_iter)
            current_epoch += 1
            n = 1

        
        optimizer.zero_grad()

        # get inputs
        rgb_img, thm_img, masks = current_batch
        rgb_img, thm_img, masks = rgb_img.cuda(), thm_img.cuda(), masks.cuda()
        n+=1

        # send through model (returns losses_dict)
        # losses_dict = model(rgb_img, thm_img, alphas)
        loss = model(thm_img, rgb_img, masks)
        writer.add_scalar(f"train loss", loss, i)

        # backprop
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # save 
        if i > 0 and i % save_step == 0:
            save_path = f"{checkpoint_dir}/chkpt_{i}.pt"
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)

    # save final model
    save_path = f"{checkpoint_dir}/final_{i+1}.pt"
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }, save_path)

def train_gan(save_dir):
    # data loaders
    train_dataset = RGBxThermalTreeCrownDataset(root='data', split='train', masks_dir='masks(30,75)')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # for logging
    writer = SummaryWriter(save_dir)
    checkpoint_dir = f'{save_dir}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = UNet(1,3)
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

    # optimizer & scheduler
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler_G = ExponentialLR(optimizer_G, gamma=0.9)
    scheduler_D = ExponentialLR(optimizer_D, gamma=0.9)

    Tensor = torch.cuda.FloatTensor 

    # loop
    n = 1
    current_epoch = 0
    max_iter = 30000
    save_step = 1000
    data_len = len(train_dataloader)
    num_epochs = max_iter / data_len
    loader_iter = iter(train_dataloader)
    loss = None
    for i in (pbar:=tqdm(range(max_iter))):
        pbar.set_description(f"{loss}")
        # get current batch (reset between epochs)
        try:
            current_batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_dataloader)
            current_batch = next(loader_iter)
            current_epoch += 1
            n = 1
            scheduler_D.step()
            scheduler_G.step()

        

        


        # get inputs
        rgb_img, thm_img, masks = current_batch
        rgb_img, thm_img, masks = rgb_img.cuda(), thm_img.cuda(), masks.cuda()
        n+=1

        

        # Configure input
        rgb_img = Variable(rgb_img)
        thm_img = Variable(thm_img)

        # Adversarial ground truths
        valid = Variable(Tensor(rgb_img.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(rgb_img.size(0), 1).fill_(0.0) , requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        out, gld = generator(thm_img, rgb_img, masks)
        for k,v in gld.items():
            writer.add_scalar(f"loss_{k}", v, i)
        gen_loss = gld['tv'] + gld['perceptual'] + gld['content'] 
        adv_loss = 0.03 * adversarial_loss(discriminator(out), valid)
        writer.add_scalar(f"loss_adv", v, i)
        gen_loss += adv_loss

        gen_loss.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        
        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(rgb_img)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(out.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2 
        writer.add_scalar(f"loss_adversarial", d_loss, i)


        d_loss.backward()
        optimizer_D.step()

        loss = [d_loss.item(), gen_loss.item(), '=', d_loss.item()+gen_loss.item()]

        # save 
        if i > 0 and i % save_step == 0:
            save_path = f"{checkpoint_dir}/chkpt_{i}.pt"
            torch.save({
                'epoch': i,
                'G_model_state_dict': generator.state_dict(),
                'G_optimizer_state_dict': optimizer_G.state_dict(),
                'G_scheduler_state_dict': scheduler_G.state_dict(),
                'D_model_state_dict': discriminator.state_dict(),
                'D_optimizer_state_dict': optimizer_D.state_dict(),
                'D_scheduler_state_dict': scheduler_D.state_dict(),
                }, save_path)
            
        # save image
        if i%100 == 0:
            img = out.detach().cpu().numpy().transpose((0,2,3,1))
            plt.subplot(141), plt.imshow(img[0]), plt.title(i)
            plt.subplot(142), plt.imshow(img[1]), plt.title(i)
            plt.subplot(143), plt.imshow(img[2]), plt.title(i)
            plt.subplot(144), plt.imshow(img[3]), plt.title(i)
            plt.savefig(f'{checkpoint_dir}/generated.png')
            
            


    # save final model
    save_path = f"{checkpoint_dir}/final_{i+1}.pt"
    torch.save({
                'epoch': i,
                'G_model_state_dict': generator.state_dict(),
                'G_optimizer_state_dict': optimizer_G.state_dict(),
                'G_scheduler_state_dict': scheduler_G.state_dict(),
                'D_model_state_dict': discriminator.state_dict(),
                'D_optimizer_state_dict': optimizer_D.state_dict(),
                'D_scheduler_state_dict': scheduler_D.state_dict(),
                }, save_path)




if __name__ == "__main__":
    # load deepforest retinaet, save weights
    # model = df_main.deepforest()
    # model.use_release()
    # model.cuda()
    # torch.save(model.model.state_dict(), 'df_retinanet.pt')

    # train('./output/latest')
    train_resume('./output/ablation/intensity/(30,75)')
    # train_enc_dec('./output/enc_dec/try1')
    # train_gan('./output/enc_dec/four_losses')



   





 