import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import trimesh
import skimage.measure

def calc_gradient(y, x, grad_outputs=None):
  if grad_outputs is None:
    grad_outputs = torch.ones_like(y)
  grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
  return grad

def sdf_loss(sdf_pred, coords, sdf_gt, normal_gt):
  gradient = calc_gradient(sdf_pred, coords)

  mask = sdf_gt != -1  # (B, N, 1)
  sdf_loss = sdf_pred[mask].abs().mean()

  inter_loss = torch.exp(-50 * torch.abs(sdf_pred[mask.logical_not()])).mean()
  mask = mask.squeeze(-1)
  normal_loss = (gradient[mask, :] - normal_gt[mask, :]).norm(2, dim=-1).mean()
  # option 2: SIREN uses
  #normal_loss = (1 - F.cosine_similarity(gradient[mask, :], normal_gt[mask, :], dim=-1)[..., None]).mean()
  grad_loss = (gradient[mask.logical_not(), :].norm(2, dim=-1) - 1).abs().mean()

  losses = [sdf_loss * 1000, inter_loss * 10, normal_loss * 20, grad_loss * 1]
  # option 2: SIREN
  #losses = [sdf_loss * 3e3, inter_loss * 1e2, normal_loss * 1e2, grad_loss * 5e1]

  total_loss = torch.stack(losses).sum()
  names = ['sdf', 'inter', 'normal_constraint', 'grad_constraint', 'total_train_loss']
  loss_dict = dict(zip(names, losses + [total_loss]))
  return loss_dict

class RandFourierFeature(nn.Module):
  def __init__(self, in_features, num_frequencies=128, sigma=1, scale=80, range=2.):
    super().__init__()
    self.in_features = in_features
    self.sigma = sigma
    self.scale = scale
    self.range = range # range of coordinate, e.g. range = 2 if input belongs to [-1,1], and 1 if [0,1].
    self.num_frequencies = num_frequencies
    self.out_features = self.num_frequencies * 2
    self.register_buffer('proj', torch.Tensor(in_features, num_frequencies))
    self.reset_parameters()
    print('random fourier feature params (sigma and scale): ',self.sigma,self.scale)

  def reset_parameters(self):
    with torch.no_grad():
      torch.manual_seed(123456) #sdf training is sensitive to gaussian fourier feature. so use a fixed one
      self.proj.copy_(torch.randn_like(self.proj) * (2*np.pi/self.range))

  def forward(self, coords):
    pos_enc = torch.mm(coords.flatten(start_dim=0, end_dim=1), self.proj*self.sigma)
    pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], axis=-1)
    output = pos_enc.view(coords.shape[0], coords.shape[1], self.out_features)
    if self.scale != -1:
      output *= self.scale / np.sqrt(self.num_frequencies) #the magnitude of fourier feature is np.sqrt(self.num_frequencies)
    return output

class FCLayer(nn.Module):
  def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.act = act

  def forward(self, input):
    output = self.linear(input)
    output = self.act(output)
    return output

class FINN(nn.Module):
  def __init__(self, in_features=3, out_features=1, hidden_features=256,num_layers=4, num_frequencies=128, sigma = 1, scale = 80):
    super().__init__()

    self.pos_enc = RandFourierFeature(in_features, num_frequencies = num_frequencies, sigma=sigma, scale=scale)
    self.scaling = nn.Linear(self.pos_enc.out_features, hidden_features)
    self.num_layers = num_layers
    for i in range(self.num_layers):
      if i == 0:
        in_channel = self.pos_enc.out_features
      else:
        in_channel = hidden_features
      setattr(self, f'FC_{i:d}', FCLayer(in_channel, hidden_features, nn.ReLU(inplace=True)))
    self.FC_final = FCLayer(hidden_features, out_features, nn.Identity())

  def forward(self, coords):
    output = self.pos_enc(coords)
    fx = self.scaling(output)
    for i in range(self.num_layers):
      fc = getattr(self, f'FC_{i:d}')
      output = F.normalize(fc(output), p=2, dim=-1) * fx
    output = self.FC_final(output)
    return output

class FFN(nn.Module):
  def __init__(self, in_features=3, out_features=1,hidden_features=256, num_layers=4, num_frequencies=128, sigma = 1, scale = -1):
    super().__init__()

    self.pos_enc = RandFourierFeature(in_features,num_frequencies = num_frequencies, sigma = sigma, scale=scale)
    self.num_layers = num_layers
    for i in range(self.num_layers):
      if i==0:
        in_channel = self.pos_enc.out_features
      else:
        in_channel = hidden_features
      setattr(self, f'FC_{i:d}', FCLayer(in_channel, hidden_features, nn.Softplus(beta=100)))
    self.FC_final = FCLayer(hidden_features, out_features, nn.Identity())

  def forward(self, coords):
    output = self.pos_enc(coords)
    for i in range(self.num_layers):
      fc = getattr(self, f'FC_{i:d}')
      output = fc(output)
    output = self.FC_final(output)
    return output

def get_mgrid(size, dim=2, offset=0.5, r=-1):
  coords = np.arange(0, size, dtype=np.float32)
  coords = (coords + offset) * 2 / size - 1  # [0, size] -> [-1, 1]
  output = np.meshgrid(*[coords]*dim, indexing='ij')
  output = np.stack(output[::r], -1)
  output = output.reshape(size**dim, dim)
  return output

class PointCloudLoader(Dataset):
  def __init__(self, pointcloud_path, on_surface_points):
    super().__init__()

    print("Loading point cloud")
    point_cloud = np.genfromtxt(pointcloud_path)
    print(point_cloud.shape[0], " points loaded")

    coords = point_cloud[:, :3]
    self.normals = point_cloud[:, 3:]

    # Reshape point cloud such that it lies in bounding box of (-1, 1)
    coords -= np.mean(coords, axis=0, keepdims=True)
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)

    self.coords = (coords - coord_min) / (coord_max - coord_min)
    self.coords -= 0.5
    self.coords *= 2.* 0.9 #points lie in box ~ [-1,1]*0.9

    self.on_surface_points = on_surface_points
    #print('point segments:',self.coords.shape[0] // self.on_surface_points)


  def __len__(self):
    return 1 #self.coords.shape[0] // self.on_surface_points

  def __getitem__(self, idx):
    point_cloud_size = self.coords.shape[0]

    off_surface_samples = self.on_surface_points  # **2
    total_samples = self.on_surface_points + off_surface_samples

    # Random coords
    rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

    on_surface_coords = self.coords[rand_idcs, :]
    on_surface_normals = self.normals[rand_idcs, :]

    off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
    off_surface_normals = np.ones((off_surface_samples, 3)) * -1

    sdf = np.zeros((total_samples, 1))  # on-surface = 0
    sdf[self.on_surface_points:, :] = -1  # off-surface = -1

    coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
    normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

    return torch.from_numpy(coords).float(),torch.from_numpy(sdf).float(),torch.from_numpy(normals).float()

def calc_sdf(model, N=256, max_batch=64**3):
  # generate samples
  #print('making a grid:', N,'x', N,'x', N)
  num_samples = N ** 3
  samples = get_mgrid(N, dim=3, offset=0, r=1)
  samples = torch.from_numpy(samples)
  sdf_values = torch.zeros(num_samples)

  # forward
  head = 0
  while head < num_samples:
    tail = min(head + max_batch, num_samples)
    sample_subset = samples[head:tail, :].cuda().unsqueeze(0)
    pred = model(sample_subset).squeeze().detach().cpu()
    sdf_values[head:tail] = pred
    head += max_batch
  sdf_values = sdf_values.reshape(N, N, N).numpy()
  return sdf_values

def create_mesh(epoch, model, mesh_dir, N=256, max_batch=64**3, level=0):
  filename = os.path.join(mesh_dir, '%s.ply' % ( epoch))
  tqdm.write("Epoch %s, Extract mesh: %s" % (epoch, filename))

  sdf_values = calc_sdf(model, N, max_batch)
  vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
  try:
    vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level)
  except:
    pass
  if vtx.size == 0 or faces.size == 0:
    print('Warning from marching cubes: Empty mesh!')
    return

  # normalize vtx
  voxel_size = 2.0 / N
  voxel_origin = np.array([-1, -1, -1])
  vtx = vtx * voxel_size + voxel_origin

  # save to ply
  mesh = trimesh.Trimesh(vtx, faces)
  mesh.export(filename)

def test_sdf(args):
  if not os.path.isfile(args.ckpt):
    print('error ckpt path')
    exit()

  if args.model == 'FINN':
    model = FINN(in_features=3,out_features=1, num_layers=args.num_layers, num_frequencies = args.num_frequencies, sigma = args.sigma, scale=args.scale)
  elif args.model == 'FFN':
    model = FFN(in_features=3,out_features=1, num_layers=args.num_layers, num_frequencies = args.num_frequencies, sigma = args.sigma, scale=-1)
  else:
    print('no active network, quit')
    exit()

  print(model)
  model.cuda()

  # load pretrained model if exist
  print('loading checkpoint %s' % args.ckpt)
  model.load_state_dict(torch.load(args.ckpt))

  create_mesh(args.test_file, model, './', N=args.res, max_batch=64 ** 3, level=0)

def run_sdf(filepath,args):
  print('read point cloud: ',filepath)
  pc_dataset = PointCloudLoader(filepath, args.pc_num)
  dataloader = DataLoader(pc_dataset, shuffle=True,batch_size=1, pin_memory=True, num_workers=0)

  #init network and optimizer
  if args.model == 'FINN':
    model = FINN(in_features=3,out_features=1, num_layers=args.num_layers, num_frequencies = args.num_frequencies, sigma = args.sigma, scale=args.scale)
  elif args.model == 'FFN':
    model = FFN(in_features=3,out_features=1, num_layers=args.num_layers, num_frequencies = args.num_frequencies, sigma = args.sigma, scale=-1)
  else:
    print('no active network, quit')
    exit()

  print(model)
  model.cuda()
  optim = torch.optim.Adam(lr=args.lr, params=model.parameters())

  # load pretrained model if exist
  if args.ckpt:
    print('loading checkpoint %s' % args.ckpt)
    model.load_state_dict(torch.load(args.ckpt))

  #
  filename = filepath.split('/')[-1]
  num_epochs = args.nr_epochs
  test_every_epoch = args.test_epochs
  logdir = os.path.join('logs',filename.split('.')[0]+'_'+args.model)
  writer = SummaryWriter(logdir)
  mesh_dir = os.path.join(logdir, 'mesh')
  ckpt_dir = os.path.join(logdir, 'checkpoints')
  if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
  if not os.path.exists(mesh_dir): os.makedirs(mesh_dir)
  fid = open(os.path.join(logdir, 'train_loss.csv'), 'w')
  tqdm.write("Epoch, Loss", fid)

  with tqdm(total=len(dataloader) * num_epochs) as pbar:
    for epoch in range(num_epochs):
      if epoch % test_every_epoch == 0:
        ckpt_name = os.path.join(ckpt_dir, 'model_%05d.pth' % epoch)
        torch.save(model.state_dict(), ckpt_name)
        create_mesh('%04d' % (epoch), model, mesh_dir, N=args.res, max_batch=64 ** 3, level=0)

      model.train()
      avg_loss = []
      for i, data in enumerate(dataloader):
        coords = data[0].cuda().requires_grad_()
        sdf_gt, normal_gt = data[1].cuda(), data[2].cuda()

        sdf = model(coords)

        losses = sdf_loss(sdf, coords, sdf_gt, normal_gt)
        total_loss = losses['total_train_loss']

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        for k, v in losses.items():
          writer.add_scalar(k, v.detach().cpu().item(), epoch * len(dataloader) + i)
        avg_loss.append(total_loss.detach().cpu().item())

        pbar.update(1)

        if i % 100 == 0:
          tqdm.write("Epoch %d, Total loss %0.6f" % (epoch, np.mean(avg_loss)))
          tqdm.write("Epoch %d, Total loss %0.6f" % (epoch, np.mean(avg_loss)), fid)

  ckpt_name = os.path.join(ckpt_dir, 'model_final.pth')
  torch.save(model.state_dict(), ckpt_name)
  create_mesh('%04d' % (num_epochs), model, mesh_dir, N=args.res, max_batch=64 ** 3, level=0)
  fid.close()

class Config(object):

  def __init__(self,):

    parser, self.args = self.parse()
    if self.args.gpu_ids is not None:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_ids)

  def parse(self):
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('basic')
    group.add_argument('-g', '--gpu_ids', type=str, default=None, help='gpu to use, e.g. 0  0,1,2.')
    parser.add_argument('--model', type=str, default='FINN', choices=('FINN', 'FFN'),
                        help='Model to use [\'FINN, FFN\']')

    group = parser.add_argument_group('network')
    group.add_argument('--sigma', type=float, default=1.0)
    group.add_argument('--scale', type=float, default=-1.0)
    group.add_argument('--num_frequencies', type=int, default=128, help='number of frequencies')
    group.add_argument('--num_layers', type=int, default=4, help='number of hidden layers')

    group = parser.add_argument_group('dataset')
    group.add_argument('--data', type=str, default='data', help='where data is')
    group.add_argument('--res', type=int, default=256, help='sdf volume resolution')

    group = parser.add_argument_group('training')
    group.add_argument('--nr_epochs', type=int, default=10000, help='total number of epochs to train')
    group.add_argument('--test_epochs', type=int, default=1000, help='total number of epochs to train')
    group.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    group.add_argument('--ckpt', type=str, default='', help='pretrained model')
    group.add_argument('--pc_num', type=int, default=32**3, help='num of samples in one step')

    group = parser.add_argument_group('testing')
    group.add_argument('--test_file', type=str, default='', help='save file')

    args = parser.parse_args()
    return parser, args

if __name__ == '__main__':
    args = Config().args
    if args.test_file:
      test_sdf(args)
    else:
      if os.path.isfile(args.data):
        run_sdf(args.data,args)
      else:
        print('error file path')

#(Training) ➜  python surface_reconstruct.py --data './thai_statue.xyz' --pc_num 100000 --model FINN -g 0
#(Testing) ➜  python surface_reconstruct.py --ckpt logs/thai_statue/checkpoints/model_final.pth --test_file thai_statue_finn  --model FINN -g 3 --res 1600
