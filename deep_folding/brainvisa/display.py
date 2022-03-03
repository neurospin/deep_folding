""" The aim of this script is to display saved model outputs thanks to Anatomist.
Model outputs are stored as numpy arrays.
"""

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
    vol_img = aims.Volume(img)
    a_vol_img = ana_a.toAObject(vol_img)
    vol_img.header()['voxel_size'] = [1, 1, 1]
    a_vol_img.setName(
        status +
        '_' +
        str(sub_id) +
        '_' +
        str(phase))  # display name
    a_vol_img.setChanged()
    a_vol_img.notifyObservers()

    return vol_img, a_vol_img


def main():
    """
    In the Anatomist window, for each model output, corresponding input will
    also be displayed at its left side.
    Number of columns and view (Sagittal, coronal, frontal) can be specified.
    (It's better to choose an even number for number of columns to display)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help='directory where model outputs are stored')
    args = parser.parse_args()
    root_dir = args.root_dir

    a = anatomist.Anatomist()
    # Parameter 6 corresponds to the number of columns displayed. Can be
    # changed.
    block = a.AWindowsBlock(a, 6)

    input_arr = np.load(root_dir + 'input.npy')  # Input
    output_arr = np.load(root_dir + 'output.npy')  # Model's output
    phase_arr = np.load(root_dir + 'phase.npy')  # Train or validation
    id_arr = np.load(root_dir + 'id.npy')  # Subject id
    for k in range(len(id_arr[0])):
        sub_id = id_arr[0][k]
        phase = phase_arr[0][k]
        input = input_arr[0][k]
        output = output_arr[0][k].astype(float)
        for img, entry in [(input, 'input'), (output, 'output')]:
            globals()[
                'block%s%s%s' %
                (sub_id,
                 phase,
                 entry)] = a.createWindow(
                'Sagittal',
                block=block)

            globals()[
                'img%s%s%s' %
                (sub_id, phase, entry)], globals()[
                'a_img%s%s%s' %
                (sub_id, phase, entry)] = array_to_ana(
                a, img, sub_id, phase, status=entry)

            globals()[
                'block%s%s%s' %
                (sub_id,
                 phase,
                 entry)].addObjects(
                globals()[
                    'a_img%s%s%s' %
                    (sub_id,
                     phase,
                     entry)])


if __name__ == '__main__':
    main()
    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()
