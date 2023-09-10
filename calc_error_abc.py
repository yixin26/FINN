import os
import argparse
from scipy.spatial import cKDTree
import numpy as np
import trimesh


parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, required=True)
parser.add_argument('--pred', type=str, required=True)
parser.add_argument('--point_num', type=int, default=64**3)
#parser.add_argument('--point_num', type=int, default=1000000)
args = parser.parse_args()

def calc_chamfer(filename_gt, filename_pred, point_num):
  scale = 1.0e5  # scale the result for better display
  np.random.seed(101)

  mesh_a = trimesh.load(filename_gt)
  coords = mesh_a.vertices
  # Reshape point cloud such that it lies in bounding box of (-1, 1)
  coords -= np.mean(coords, axis=0, keepdims=True)
  coord_max = np.amax(coords)
  coord_min = np.amin(coords)
  coords = (coords - coord_min) / (coord_max - coord_min)
  coords -= 0.5
  coords *= 2. * 0.9  # points lie in box ~ [-1,1]*0.9
  mesh_a.vertices = coords
  points_a, _ = trimesh.sample.sample_surface(mesh_a, point_num)

  mesh_b = trimesh.load(filename_pred)
  points_b, _ = trimesh.sample.sample_surface(mesh_b, point_num)

  kdtree_a = cKDTree(points_a)
  dist_a, _ = kdtree_a.query(points_b)
  chamfer_a = np.mean(np.square(dist_a)) * scale

  kdtree_b = cKDTree(points_b)
  dist_b, _ = kdtree_b.query(points_a)
  chamfer_b = np.mean(np.square(dist_b)) * scale
  return chamfer_a, chamfer_b

test_ffn = True
test_finn =True

if os.path.isfile(args.gt):
  filename_gt = args.gt
  filename_pred = args.pred
  point_num = args.point_num

  chamfer_a, chamfer_b = calc_chamfer(filename_gt, filename_pred, point_num)
  result = '{}, {}, {}, {:.4f}, {:.4f}, {:.4f}\n'.format(
      os.path.dirname(filename_gt),
      os.path.basename(filename_pred),
      point_num, chamfer_a, chamfer_b, chamfer_a + chamfer_b)
  print(result)
elif os.path.isdir(args.gt):
  files = os.listdir(args.gt)
  files.sort()
  print(len(files),'shapes')
  num = 100
  cd_ffn_avg = cd_finn_avg = .0
  count = 0
  for filename in files:
    filename_gt = os.path.join(args.gt, filename)
    print(filename)
    cd_ffn = cd_finn = .0

    if test_ffn:
      print('FFN:')
      filename_pred = os.path.join(args.pred, filename.split('.')[0]+'_FFN','mesh','10000.ply')
      if not os.path.isfile(filename_pred):
        print(filename_pred, 'does not exist!')
        continue

      point_num = args.point_num

      chamfer_a, chamfer_b = calc_chamfer(filename_gt, filename_pred, point_num)
      result = '{}, {}, {}, {:.4f}, {:.4f}, {:.4f}\n'.format(
          os.path.dirname(filename_gt),
          os.path.basename(filename_pred),
          point_num, chamfer_a, chamfer_b, chamfer_a + chamfer_b)
      print(result)
      cd_ffn = chamfer_a + chamfer_b

    if test_finn:
      print('FINN:')
      filename_pred = os.path.join(args.pred, filename.split('.')[0] + '_FINN', 'mesh', '10000.ply')
      if not os.path.isfile(filename_pred):
        print(filename_pred, 'does not exist!')
        continue

      point_num = args.point_num

      chamfer_a, chamfer_b = calc_chamfer(filename_gt, filename_pred, point_num)
      result = '{}, {}, {}, {:.4f}, {:.4f}, {:.4f}\n'.format(
        os.path.dirname(filename_gt),
        os.path.basename(filename_pred),
        point_num, chamfer_a, chamfer_b, chamfer_a + chamfer_b)
      print(result)
      cd_finn = chamfer_a + chamfer_b
    if cd_ffn > 100 or cd_finn > 100:
      continue

    cd_ffn_avg += cd_ffn
    cd_finn_avg += cd_finn

    if cd_ffn <= cd_finn:
      count += 1

    num -=1
    if num < 0: break

print("FFN:", cd_ffn_avg/100., "  FINN:", cd_finn_avg/100.)
print(count)
#python calc_error.py --gt ./abc/10k/train/2048/shape_name.obj --pred ./logs/shape_name_FINN/mesh/10000.ply
