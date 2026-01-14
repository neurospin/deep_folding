#!/usr/bin/env python

import anatomist.direct.api as ana
from soma import aims
import os.path as osp
from soma.qt_gui.qt_backend import Qt
from deep_folding import config


icbm_mesh_dir_fallback = '/neurospin/dico/data/bv_databases/templates/morphologist_templates/icbm152/mni_icbm152_nlin_asym_09c/t1mri/default_acquisition/default_analysis/segmentation/mesh'
regions_graph_pat = '%(side)sregions_model_1.arg'
regions1 = [
    'S.F.median-S.F.pol.tr.-S.F.sup.',
    'S.F.marginal-S.F.inf.ant.',
    'S.Or.-S.Olf.',
    'S.Pe.C.',
    'S.C.-sylv.',
    'S.Po.C.',
    'F.C.L.p.-subsc.-F.C.L.a.-INSULA.',
    'S.T.s.br.',
    'S.T.s.',
    'S.T.i.-S.O.T.lat.',
    'OCCIPITAL',
    'F.Coll.-S.Rh.',
    'F.P.O.-S.Cu.-Sc.Cal.',
    'S.F.int.-F.C.M.ant.',
    'S.Call.',
    'F.C.M.post.-S.p.C.',
    'S.s.P.-S.Pa.int.',
    'Lobule_parietal_sup.',
]


def other_regions(nom, regions1):
    regions1 = set(regions1)
    other = []
    for node in nom.children()[0].children()[0].children():
        name = node['name'].rsplit('_', 1)[0]
        if name not in regions1:
            other.append(name)
    return other


def display_champo_regions():
    a = ana.Anatomist()
    root = config.config().get_champollion_data_root_dir()
    regions_graph_dir = f"{root}/mask/2mm/regions/meshes"
    nom = aims.read(aims.carto.Paths.findResourceFile(
        'nomenclature/hierarchy/champollion_v1.hie'))
    regions2 = other_regions(nom, regions1)
    anom = a.toAObject(nom)
    l_reg_graph = a.loadObject(osp.join(regions_graph_dir,
                                        regions_graph_pat % {'side': 'L'}))
    r_reg_graph = a.loadObject(osp.join(regions_graph_dir,
                                        regions_graph_pat % {'side': 'R'}))
    l_reg_graph.applyBuiltinReferential()
    r_reg_graph.applyBuiltinReferential()
    reg_graphs = (l_reg_graph, r_reg_graph)

    # load ICBM brain meshes
    icbm_mesh_dir = aims.carto.Paths.findResourceFile(
        'disco_templates_hbp_morpho/icbm152/mni_icbm152_nlin_asym_09c/t1mri/'
        'default_acquisition/default_analysis/segmentation/mesh', 'disco')
    if icbm_mesh_dir is None:
        # fallback to hard-coded path in neurospin
        icbm_mesh_dir = icbm_mesh_dir_fallback
    l_mesh = a.loadObject(osp.join(icbm_mesh_dir,
                                   'mni_icbm152_nlin_asym_09c_Lhemi.gii'))
    r_mesh = a.loadObject(osp.join(icbm_mesh_dir,
                                   'mni_icbm152_nlin_asym_09c_Rhemi.gii'))
    l_mesh.setMaterial(diffuse=[0.8, 0.8, 0.8, 0.37])
    l_mesh.applyBuiltinReferential()
    r_mesh.setMaterial(diffuse=[0.8, 0.8, 0.8, 0.37])
    r_mesh.applyBuiltinReferential()
    meshes = (l_mesh, r_mesh)

    wins = []
    block = a.createWindowsBlock(nbRows=2)
    group = 0
    orients = ((0.5, 0.5, 0.5, 0.5), (0.5, -0.5, -0.5, 0.5))
    app = Qt.QApplication.instance()

    for side, reg_graph, mesh, orient_i in zip(
            ('left', 'right'), reg_graphs, meshes, (0, 1)):
        for orient in (orients[orient_i], orients[1 - orient_i]):
            for regions in (regions1, regions2):
                w = a.createWindow('3D', block=block, no_decoration=True)
                w.windowConfig(cursor_visibility=0,
                               polygons_depth_sorting=True)
                a.execute('LinkWindows', windows=[w], group=group)
                wins.append(w)
                w.addObjects([reg_graph, mesh], add_graph_nodes=False)
                w.setReferential(reg_graph.referential)
                sel_regions = ' '.join(f'{r}_{side}' for r in regions)
                a.execute('SelectByNomenclature', nomenclature=anom,
                          names=sel_regions, group=group)
                a.execute('SelectByNomenclature', nomenclature=anom,
                          names=sel_regions, modifiers='toggle', group=group)
                w.camera(view_quaternion=orient)
                app.processEvents()
                Qt.QTimer.singleShot(1000, w.getSliceSlider().parent().hide)

                group += 1
                del w

    app.exec()
    del wins, block, reg_graphs, l_reg_graph, r_reg_graph


if __name__ == '__main__':
    display_champo_regions()
