import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

from jittor.misc import save_image
import jittor as jt
from jittor import nn, optim
from jittor import grad
# from jittor.utils.data import DataLoader
from jittor import  transform

from dataset import ColorSymbol
from model import StyledGenerator, Discriminator

jt.flags.use_cuda  = 1
jt.flags.log_silent = 1

os.makedirs('sample', exist_ok=True)
os.makedirs('checkpoint', exist_ok=True)


def requires_grad(model, flag=True):
        for p in model.parameters():
                    p.requires_grad = flag

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k] * decay + (1 - decay) *par2[k].detach())

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def train(args, dataset, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    dataset.resolution = resolution
    loader = dataset
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(20_001))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        d_optimizer.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))
        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step
            dataset.resolution = resolution
            loader = dataset
            data_loader = iter(loader)

            if (jt.rank == 0):

                print("save model")
                jt.save(
                    {
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        'g_running': g_running.state_dict(),
                    },
                    f'checkpoint/train_step-{ckpt_step}.model',
                )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)


        used_sample += real_image.shape[0]

        b_size = real_image.size(0)

        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        d_optimizer.backward(-real_predict)


        gen_in1, gen_in2 = jt.randn(2, b_size, code_size).chunk(
            2, 0
        )
        gen_in1 = gen_in1.squeeze(0)
        gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        fake_predict = fake_predict.mean()
        d_optimizer.backward(fake_predict)

        eps = jt.rand(b_size, 1, 1, 1)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            hat_predict.sum(), x_hat
        )
        grad_penalty = (
            (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()
        grad_penalty = 10 * grad_penalty
        d_optimizer.backward(grad_penalty)
        if i%10 == 0:
            grad_loss_val = grad_penalty.item()
            disc_loss_val = (-real_predict + fake_predict).item()


        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            g_optimizer.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()

            if i%10 == 0:
                gen_loss_val = loss.item()

            g_optimizer.backward(loss)
            g_optimizer.step()
            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)
            
        if jt.rank == 0 and (i + 1) % 100 == 0:  
            print("save img")
            images = []

            gen_i, gen_j =  (10, 5)

            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            jt.randn(gen_j, code_size), step=step, alpha=alpha
                        ).data
                    )

            save_image(
                jt.concat(images, 0),
                f'sample/{str(20000 + i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 2000 == 0 and jt.rank==0:
            print("save_running")
            jt.save(
                g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
            )

        if jt.rank==0:
            state_msg = (
                f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
                f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
            )
            pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512 
    batch_size = 32
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

   # parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=3200,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=128, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--data_root', default='color_symbol_7k', type=str
    )
  # 为了代码简洁就用wgan-gp了
  #  parser.add_argument(
  #      '--loss',
  #      type=str,
  #      default='wgan-gp',
  #      choices=['wgan-gp', 'r1'],
  #      help='class of gan loss',
  #  )

    args = parser.parse_args()

    generator = StyledGenerator(code_size)
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    g_running = StyledGenerator(code_size)
    g_running.eval()

    g_optimizer = optim.Adam(
        generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    if args.ckpt is not None:
        ckpt = jt.load(args.ckpt)

        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transforms = transform.Compose(
        [
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5], std=[0.5]) 
        ]
    )

    dataset = ColorSymbol(shuffle=True, data_root=args.data_root, batch_size = batch_size, transform=transforms)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

   # args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, generator, discriminator)

