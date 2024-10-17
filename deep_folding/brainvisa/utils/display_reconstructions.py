""" The aim of this script is to display saved model outputs with Anatomist.
Model outputs are stored as numpy arrays.
"""
import argparse
import anatomist.api as anatomist
import dico_toolbox as dtx
import numpy as np
from soma import aims


_VOXEL_SIZE = 1
_BUCKET = True


def array_to_ana(ana_a, img, sub_id, phase, status, bucket, vs):
    """
    Transforms output tensor into volume readable by Anatomist and defines
    name of the volume displayed in Anatomist window.
    Returns volume displayable by Anatomist either on the skeleton mode,
    that can be viewed in consecutive 2D slices, or on 3D buckets.
    """
    if bucket:
        vol_img = img
        vol_img.header()['voxel_size'] = list(vs)
    else:
        vol_img = aims.Volume(img)

    a_vol_img = ana_a.toAObject(vol_img)
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
    parser.add_argument(
        "--voxel_size",
        type=int,
        default=_VOXEL_SIZE,
        help='voxel size for visualizing buckets')
    parser.add_argument(
        "--bucket",
        type=bool,
        default=_BUCKET,
        help='True if bucket visualization, else skeleton')
    args = parser.parse_args()
    root_dir = args.root_dir
    bucket = args.bucket
    vs = args.voxel_size
    vox_size = (vs, vs, vs)

    a = anatomist.Anatomist()
    # Parameter 6 corresponds to the number of columns displayed. Can be
    # changed.
    block = a.AWindowsBlock(a, 12)

    input_arr = np.load(root_dir + 'input.npy')  # Input
    output_arr = np.load(root_dir + 'output.npy')  # Model's output
    phase_arr = np.load(root_dir + 'phase.npy')  # Train or validation
    id_arr = np.load(root_dir + 'id.npy')  # Subject id
    view = 'Axial'
    for k in range(len(input_arr[0])):
        sub_id = id_arr[0][k]
        phase = phase_arr[0][k]
        input = input_arr[0][k]
        print(input.shape)
        output = output_arr[0][k].astype(float)
        if bucket:
            view = '3D'
            if len(np.unique(input)) == 2:
                input = dtx.convert.volume_to_bucketMap_aims(
                    input, voxel_size=vox_size)
                output = dtx.convert.volume_to_bucketMap_aims(
                    output, voxel_size=vox_size)
            else:
                input[input > 0.5] = 1
                input[input <= 0.5] = 0
                output[output > 0.5] = 1
                output[output <= 0.5] = 0
                input = dtx.convert.volume_to_bucketMap_aims(
                    input, voxel_size=vox_size)
                output = dtx.convert.volume_to_bucketMap_aims(
                    output, voxel_size=vox_size)

        for img, entry in [(input, 'input'), (output, 'output')]:
            globals()[
                'block%s%s%s' %
                (sub_id,
                 phase,
                 entry)] = a.createWindow(
                view,
                block=block)

            globals()[
                'img%s%s%s' %
                (sub_id, phase, entry)], globals()[
                'a_img%s%s%s' %
                (sub_id, phase, entry)] = array_to_ana(
                    a, img, sub_id, phase,
                    status=entry, bucket=bucket, vs=vox_size)

            globals()['block%s%s%s' % (sub_id, phase, entry)].addObjects(
                globals()['a_img%s%s%s' %
                          (sub_id, phase, entry)])


if __name__ == '__main__':
    main()
    from soma.qt_gui.qt_backend import Qt
    Qt.qApp.exec_()
