from .resnet import *
from .query2label import Query2Label
query2label = Query2Label
from .tresnet import tresnetm, tresnetl, tresnetxl, tresnetl_21k

from .tresnet2 import tresnetl as tresnetl_v2
from .swin_transformer import build_swin_transformer
