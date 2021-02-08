import anatomist.api as anatomist
from soma import aims
import numpy as np
import argparse


def array_to_ana(ana_a, img, sub_id, phase, status):
    """
    Transforms output tensor into volume readable by Anatomist and defines
    name of the volume displayed in Anatomist window.
    Returns volume displayable by Anatomist
    """
    #a = anatomist.Anatomist()

    vol_img = aims.Volume(img)
    a_vol_img = ana_a.toAObject(vol_img)
    vol_img.header()['voxel_size'] = [1, 1, 1]
    a_vol_img.setName(status+'_'+ str(sub_id)+'_'+str(phase))
    a_vol_img.setChanged()
    a_vol_img.notifyObservers()

    return vol_img, a_vol_img


def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help='date of experiment to display')
    parser.add_argument("--loss_type", type=str, help='Loss to compute , can be "L1", "L2", "SSIM"')
    parser.add_argument("--skeleton", type=bool, help='whether data is skeleton or not')
    parser.add_argument("--group", type=str, help='which group to display')
    args = parser.parse_args()
    date = args.date
    loss_type = args.loss_type
    skeleton = args.skeleton
    group = args.group

    a = anatomist.Anatomist()
    block = a.AWindowsBlock(a, 12)
    block = a.AWindowsBlock(a, 6)


    root_dir = '/neurospin/dico/lguillon/mic21/left_hemi_skeleton_101220_CrossEnt_1_3classes/'
    #root_dir = '/neurospin/dico/lguillon/mic21/left_hemi_norm_spm_101220_L2_1/'
    #root_dir = '/neurospin/dico/lguillon/data/vae/040720_skel/'
    root_dir = '/neurospin/dico/lguillon/mic21/test_oc/left_hemi_skeleton_101220_CrossEnt_1_3classes/'
    root_dir = '/neurospin/dico/lguillon/mic21/gridsearch/left_hemi_skeleton_141220_CrossEnt_1_2classes/'
    root_dir = '/neurospin/dico/lguillon/mic21/anomalies_set/vae6/left_hemi_skeleton_080221_CrossEnt_1_2classes/'
    #root_dir = '/neurospin/dico/lguillon/mic21/anomalies_set/left_hemi_skeleton_080221_CrossEnt_1_2classes/'


    if test:
        for pop in ['ASD', 'Controls']:
            input_arr = np.load(root_dir+pop+'_input.npy')
            output_arr = np.load(root_dir + pop +'_output.npy')
            phase_arr = np.load(root_dir + pop + '_phase.npy')
            id_arr = np.load(root_dir + pop +'_id.npy')

            #for k in range(len(id_arr[0])):
            for k in range(20):
                print(id_arr[0][k])
                sub_id = id_arr[0][k]
                phase = phase_arr[0][k]
                input = input_arr[0][k]
                output = output_arr[0][k].astype(float)
                for img, entry in [(input, 'input'), (output, 'output')]:
                    print(entry)
                    globals()['block%s%s%s' % (sub_id, phase, entry)] = a.createWindow('Sagittal', block=block)

                    globals()['img%s%s%s' % (sub_id, phase, entry)], globals()['a_img%s%s%s' % (sub_id,
                                phase, entry)] = array_to_ana(a, img, sub_id, phase, status=entry)

                    globals()['block%s%s%s' % (sub_id, phase, entry)].addObjects(globals()['a_img%s%s%s' % (sub_id, phase, entry)])


    else:
        input_arr = np.load(root_dir+'input.npy')
        output_arr = np.load(root_dir+'output.npy')
        phase_arr = np.load(root_dir+'phase.npy')
        id_arr = np.load(root_dir+'id.npy')
        print(id_arr)
        for k in range(len(id_arr[0])):
            print(id_arr[0][k])
            sub_id = id_arr[0][k]
            phase = phase_arr[0][k]
            input = input_arr[0][k]
            output = output_arr[0][k].astype(float)
            for img, entry in [(input, 'input'), (output, 'output')]:
                globals()['block%s%s%s' % (sub_id, phase, entry)] = a.createWindow('Sagittal', block=block)

                globals()['img%s%s%s' % (sub_id, phase, entry)], globals()['a_img%s%s%s' % (sub_id,
                                phase, entry)] = array_to_ana(a, img, sub_id, phase, status=entry)

                globals()['block%s%s%s' % (sub_id, phase, entry)].addObjects(globals()['a_img%s%s%s' % (sub_id, phase, entry)])



if __name__ == '__main__':
    test = False
    main()
    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()
