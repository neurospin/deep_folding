#!/usr/bin/env/python

from deep_folding import config
import json
from soma import aims, aimsalgo
from soma.aimsalgo import sulcitools
import os


resolution = "2mm"
champo_version = "V1"
regions_dir_pat = "mask/%(resolution)s/regions"
regions_def_file_pat = "sulci_regions_champollion_%(version)s.json"


def read_spams(champo_root=None, resolution=resolution,
               champo_version=champo_version, smooth=3.):
    if champo_root is None:
        champo_root = config.config().get_champollion_data_root_dir()
    regions_def_file \
        = f"{champo_root}/{regions_def_file_pat}" % {"version": champo_version}
    with open(regions_def_file) as f:
        regions_def = json.load(f)
    short_sides = {"left": "L", "right": "R"}
    spam = None
    labels = {}
    index = 0
    nreg = len(regions_def["brain"])

    nom = aims.read(
        aims.carto.Paths.findResourceFile(
            'nomenclature/hierarchy/champollion_v1.hie'))

    gs = aimsalgo.Gaussian3DSmoothing_S16(smooth, smooth, smooth)

    for region, reg_def in regions_def["brain"].items():
        print(region)
        side = region.rsplit("_", 1)[1]
        sside = short_sides[side]
        reg_file = f"{champo_root}/{regions_dir_pat}/{sside}/{region}.nii.gz" \
            % {"resolution": resolution}
        vol = aims.read(reg_file, border=1)

        if smooth != 0.:
            vol = gs.doit(vol)

        if spam is None:
            spam = aims.Volume(vol.shape[:3] + (nreg, ), dtype=vol.np.dtype)
            spam.copyHeaderFrom(vol.header())
        spam[:, :, :, index] = vol[:, :, :, 0]

        try:
            color = nom.find_color(region)
        except Exception:
            color = [1., 0., 0., 1.]
        labels[index] = {"Label": region, "RGB": color}

        index += 1

    spam.header()["GIFTI_labels_table"] = labels
    return spam


def save_graphs_and_meshes(spam, graphs, allmeshes, out_dir):
    dref = aims.StandardReferentials.mniTemplateReferentialID()
    try:
        try:
            icbmi = spam.header()['referentials'].index(dref)
        except ValueError:
            dref = aims.StandardReferentials.mniTemplateReferential()
            icbmi = spam.header()['referentials'].index(dref)
        icbm = aims.AffineTransformation3d(spam.header()['transformations'][
            icbmi])
    except ValueError:
        dref = aims.StandardReferentials.acPcReferentialID()
        try:
            tali = spam.header()['referentials'].index(dref)
        except ValueError:
            dref = aims.StandardReferentials.acPcReferential()
            tali = spam.header()['referentials'].index(dref)
        tal = aims.AffineTransformation3d(spam.header()['transformations'][
            tali])
        icbm = aims.StandardReferentials.talairachToICBM() * tal

    for index, (th, amesh) in enumerate(allmeshes.items()):
        for label, mesh in amesh.items():
            # dup mesh
            mesh = aims.AimsSurfaceTriangle(mesh)
            mesh.header()['referential'] \
                = aims.StandardReferentials.mniTemplateReferentialID()
            aims.SurfaceManip.meshTransform(mesh, icbm)
            aims.write(mesh, f'{out_dir}/regions_{label}_{index}.gii')

    lsides = {'L': 'left', 'R': 'right'}
    for index, (th, graphl) in enumerate(graphs.items()):
        for side, graph in zip(['L', 'R'], graphl):
            aims.write(graph, f'{out_dir}/{side}regions_model_{index}.arg')
            lside = lsides[side]
            aims.write(
                [m for l, m in allmeshes[th].items()
                 if l.endswith(lside)],
                f'{out_dir}/{side}regions_model_{index}.glb')


if __name__ == "__main__":
    champo_root = config.config().get_champollion_data_root_dir()
    spams = read_spams(champo_root, resolution, champo_version)
    print('spams:', spams.shape)
    print(spams.header())
    graphs, allmeshes = sulcitools.spam_to_graphs(spams)
    print('graphs:', graphs)
    meshdir = f"{champo_root}/{regions_dir_pat}/meshes" \
        % {"resolution": resolution}
    os.makedirs(meshdir, exist_ok=True)
    save_graphs_and_meshes(spams, graphs, allmeshes, meshdir)
