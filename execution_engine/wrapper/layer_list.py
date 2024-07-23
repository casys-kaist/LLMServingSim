import glob
import re

class LayerList:
    # class that makes a list of layer names
    def __init__(self, layers_path):
        # path that have all layers 
        self.path = layers_path
        # each layer's json file path
        self.layers_path = sorted(glob.glob(layers_path+'/layer*/*.json'), key=lambda x : int(re.search('\d+', re.search('layer\d+', x).group(0)).group(0)))
        self.layers_name = []
        for l in self.layers_path: # get the layer name in the path
            for n in l.split('/'):
                if "layer" in n:
                    self.layers_name.append(n)
