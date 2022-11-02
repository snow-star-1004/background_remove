import os
import PIL.Image
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.u2net import U2NET
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

u2net = U2NET(device='cpu',
              batch_size=2)

fba = FBAMatting(device='cpu',
                 input_tensor_size=2048,
                 batch_size=2)

trimap = TrimapGenerator()

preprocessing = PreprocessingStub()

postprocessing = MattingMethod(matting_module=fba,
                               trimap_generator=trimap,
                               device='cpu')

interface = Interface(pre_pipe=preprocessing,
                      post_pipe=postprocessing,
                      seg_pipe=u2net)
path = './input'
files = os.listdir(path)
for f in files:
    input_path = './input/'+f
    output_path = 'output/'+f+".png"
    
    image = PIL.Image.open(input_path)
    cat_wo_bg = interface([image])[0]
    cat_wo_bg.save(output_path) 
    










    