#!/usr/bin/env python

import anatomist.direct.api as ana
from soma import aims
import json
from soma.qt_gui.qt_backend import Qt
import sys

# launch first "Check SPAM models installation"

def load_regions(filename):
    with open(filename) as f:
        regions = json.load(f)
    return regions


def load_sulci_model(model_version='2019'):
    hpath = aims.carto.Paths.findResourceFile(
        'nomenclature/hierarchy/sulcal_root_colors.hie')
    if model_version == '2008':
        reg_type = 'global_registered_spam'
    else:
        reg_type = 'talairach_spam'
    lpath = aims.carto.Paths.findResourceFile(
        f'models/models_{model_version}/descriptive_models/segments'
        f'/{reg_type}_left/meshes/Lspam_model_meshes_1.arg')
    rpath = aims.carto.Paths.findResourceFile(
        f'models/models_{model_version}/descriptive_models/segments'
        f'/{reg_type}_right/meshes/Rspam_model_meshes_1.arg')
    a = ana.Anatomist()
    hie = a.loadObject(hpath)
    l_model = a.loadObject(lpath)
    r_model = a.loadObject(rpath)
    return hie, l_model, r_model


class RegionWidget(Qt.QComboBox):
    def __init__(self, regions, models):
        super().__init__()
        self.regions = regions
        self.models = models
        wid = self
        wid.addItems(sorted(regions['brain'].keys()))
        wid.textActivated.connect(self.select_region)

    def select_region(self, region):
        # print('region:', region)
        rnames = list(self.regions['brain'][region].keys())
        a = ana.Anatomist()
        a.execute('SelectByNomenclature', nomenclature=self.models[0],
                  names=' '.join(rnames), modifiers='add') # modifiers='set' or 'add')


if __name__ == '__main__':
    region_path = aims.carto.Paths.findResourceFile(
        'nomenclature/translation/sulci_regions_overlap.json')
    if len(sys.argv) > 1:
        region_path = sys.argv[1]

    regions = load_regions(region_path)

    a = ana.Anatomist()
    models = load_sulci_model()

    a.execute('SetMaterial', objects=models[1:], diffuse=[1., 1., 1., 0.3],  # you can set opacity here
              selectable_mode='always_selectable')
    a.execute('GraphParams', selection_color=[229, 51, 38, 255, 0])

    w = a.createWindow('3D')
    reg_com = RegionWidget(regions, models)
    reg_com.setParent(w.getInternalRep().centralWidget())

    w.getInternalRep().centralWidget().layout().insertWidget(0, reg_com)
    w.addObjects(models[1:])

    a.execute('SelectByNomenclature', nomenclature=models[0], names='unknown',
              modifiers='remove')

    qapp = Qt.QApplication.instance()
    qapp.exec()

    del reg_com, w, regions, models

