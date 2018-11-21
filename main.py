# coding:utf8
import os
import torch as t
import torchvision as tv
import tqdm
from model import Generator, Discriminator
from visualize import plot_loss,plot_result,create_gif

class Config(object):
    data_path = 'data/'  # path for data
    num_workers = 4  # num_GPUs
    image_size = 96  # size of image
    batch_size = 64
    max_epoch = 155
    lr1 = 2e-4  # lr for generator
    lr2 = 2e-4  # lr for discriminator
    beta1 = 0.5  # Adam optimizer beta1 parameter
    gpu = True  # use GPU?
    nz = 100  # Noise dimension
    ngf = 64  # Generator feature map
    ndf = 64  # Discriminator feature map

    save_path = 'imgs/'  # path to save results

    plot_every = 1  # 
    print_every = 5

    d_every = 1  # train discriminator once per batch
    g_every = 5  # train generator 5 times per batch
    save_every = 10  # save every 10 epochs
    netd_path = None  # 'checkpoints/netd_.pth' 
    netg_path = None  # 'checkpoints/netg_211.pth'

opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device=t.device('cuda') if opt.gpu else t.device('cpu')

    # transforms
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )

    # Generator and Discriminator network
    G, D = Generator(opt), Discriminator(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        D.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        G.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    D.to(device)
    G.to(device)


    # optimizers and criterion
    optimizer_g = t.optim.Adam(G.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(D.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # true_label=1，fake_label=0, fixed noise
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    losses = []

    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
        
            real_img = img.to(device)
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
            if ii % opt.d_every == 0:
                # clear any gradients
                optimizer_d.zero_grad()
                ## Train with real images
                output = D(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                ## Train with fake images
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = G(noises).detach()  # generate fake image
                output = D(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()
                # add fake + real loss
                error_d = error_d_fake + error_d_real
            
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
            if ii % opt.g_every == 0:
                # clear any gradients
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = G(noises)
                output = D(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()

            if ii % opt.print_every == 0:
                # append losses
                losses.append((error_d.item(), error_g.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, opt.max_epoch, error_d.item(), error_g.item()))

            
            if  ii % opt.plot_every == 0:
                ## generate samples
                fix_fake_imgs = G(fix_noises)
        
        # for plotting result and loss
        plot_result(G, noises, 96, epoch + 1, './plots')
        plot_loss(losses, epoch + 1, opt.max_epoch, './plots')


        if (epoch) % opt.save_every == 0:

            # save generated images
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,range=(-1, 1))
            t.save(D.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            t.save(G.state_dict(), 'checkpoints/netg_%s.pth' % epoch)


if __name__ == '__main__':

    # start training
    train()
    # create gif for samples and loss
    create_gif(opt.max_epoch, './plots')
