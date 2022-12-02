import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh

class ClassificationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, opt=self.opt, hold_history=False, export_folder=self.opt.export_folder)
        meta = {'mesh': mesh, 'label': label}


        # get Edge Centroid Coord
        
        edge_features = mesh.extract_features() #obtian edge_features
        edges = mesh.get_edges() #Obtain the edges
        vs = mesh.get_v() #Obtain the vertices

        # print(edge_features[5][3])

        count = 0

        for v in edges:

          locCenX = 0
          locCenY = 0
          locCenZ = 0

          #Obtain Vertices Ids
          v1Id = v[0]
          v2Id = v[1]

          #Compute Centroid Coord
          locCenX = (vs[v1Id][0] + vs[v2Id][0])/2
          locCenY = (vs[v1Id][1] + vs[v2Id][1])/2
          locCenZ = (vs[v1Id][2] + vs[v2Id][2])/2

          #Assign into the model
          edge_features[0][count] = locCenX
          edge_features[1][count] = locCenY
          edge_features[2][count] = locCenZ

          count += 1


          #print(v1Id,v2Id)


        #print(len(edges))

        #print(edges[43][0],edges[32][1])
        
        # print(vs[0],vs[0][2])
        # edge_features = pad(edge_features, self.opt.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes
