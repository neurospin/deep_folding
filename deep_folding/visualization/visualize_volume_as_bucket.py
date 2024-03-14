#!python



import anatomist.direct.api as ana
from soma import aims
import numpy as np
from soma.qt_gui.qt_backend import Qt
import sys
import dico_toolbox as dtx

def convert_volume_to_bucket(vol):
    """Converts volume to bucket

    Args:
        arr: numpy array
    """
    c = aims.Converter_rc_ptr_Volume_S16_BucketMap_VOID()
    bucket_map = c(vol)
    bucket = bucket_map[0]
    bucket = np.array([bucket.keys()[k].list()
                      for k in range(len(bucket.keys()))])
    return bucket_map, bucket



if __name__ == '__main__':

    a = ana.Anatomist('-b')
    
    volume_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE/mask/Rlabels/sub-1000021_cropped_foldlabel.nii.gz"
    select_min = 6000
    select_max = 6999
        
    # Input analysis
    if len(sys.argv) == 2:
        volume_path = sys.argv[1]
    elif len(sys.argv) > 2:
        volume_path = sys.argv[1]
        select_min = int(sys.argv[2])
        select_max = int(sys.argv[3])
    else:
        # volume_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE/mask/Rcrops/sub-1000021_cropped_skeleton.nii.gz"
        # select_min = 35
        # select_max = 35
    
    rgb = [0.8, 0., 0.]
    
    vol = aims.read(volume_path)
    
    vol1 = aims.Volume(vol)
    vol1.np[(vol1.np>=select_min) & (vol1.np<=select_max)] = 0
    bucket_map, _ = convert_volume_to_bucket(vol1)
    
    vol2 = aims.Volume(vol)
    vol2.np[(vol2.np<select_min) | (vol2.np>select_max)] = 0
    bucket_map2, _ = convert_volume_to_bucket(vol2)
    
    # Visualization with anatomist
    
    abucket = a.toAObject(bucket_map)
    abucket.setName("without top")
    abucket2 = a.toAObject(bucket_map2)
    abucket2.setName("without top")

    win = a.createWindow('3D')
    win.windowConfig(cursor_visibility=0)
    
    win.addObjects([abucket, abucket2])
    
    a.execute('SetMaterial', objects=[abucket], diffuse=[0.8, 0.8, 0.8, 0.7])
    a.execute('SetMaterial', objects=[abucket2], diffuse=rgb +[1.])
    
    qapp = Qt.QApplication.instance()
    qapp.exec()
    
    # Finalization
    del vol, bucket_map, abucket, abucket2, win, a


