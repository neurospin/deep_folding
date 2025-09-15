#!/usr/bin/env python

import anatomist.api as ana
from .ana_sulcal_roi_view import load_sulci_model, load_regions
from soma import aims
import pandas as pd
import json
import numpy as np
import os.path as osp
import argparse


# constants
SQ2_2 = np.sqrt(2.) / 2.
""" pi / 2, used in quaternions for convenience """
VIEW_SIZE_FACOR = 1.
""" modifiable size factor for default snapshot sizes """
DEFAULT_VIEW_SIZES = [
    (600, 600),  # front
    (600, 600),  # back
    (600, 800),  # top
    (600, 800),  # bottom
    (800, 600),  # right
    (800, 600),  # left
]
""" default snapshot sizes for each orientation """
DEFAULT_QUATERNIONS = [
    (SQ2_2, 0., 0., SQ2_2),  # front
    (0., SQ2_2, SQ2_2, 0.),  # back
    (0., 0., 1., 0.),  # top
    (1., 0., 0., 0.),  # bottom
    (0.5, -0.5, -0.5, 0.5),  # right
    (0.5, 0.5, 0.5, 0.5),  # left
]
""" default orientations list """


def read_regions_data(source, column=0):
    """ Get regions data from a file, or a tructured object.

    Parameters
    ----------
    source: str or dict or numpy array or list or pandas dataframe
        If the source is a string, it is assumed to be a filename. CSV or JSON
        formats are recognzed.
    column: int
        column in the source table to be mapped

    Returns:
    --------
    data: dict
        keys are regions names, values are the scalar values to be mapped on
        the glassbrain
    """
    if isinstance(source, dict):
        return source

    if isinstance(source, (np.ndarray, list)):
        return {item[0]: item[1] for item in source}

    if isinstance(source, pd.DataFrame):
        col = source.columns[column]
        return {ind: source.iloc[i][col]
                for i, ind in enumerate(source.index)}

    if source.endswith('.json'):
        with open(source) as f:
            data = json.load(f)
        return data

    if source.endswith('.csv'):
        df = pd.read_csv(source, index_col=0)
        col = df.columns[column]
        return {ind: df.iloc[i][col]
                for i, ind in enumerate(df.index)}

    raise ValueError(
        f'{source} is not recognized as a readable file or dataset')


def region_to_sulci_mapping(regions):
    """ returns a dict {sulcus: region}

    Each sulcus is assigned a region which represents it "the best".
    The iterative selection algorithm is a bit complex:
    We pich a sulcus amongst the "smallest" regions. Amongst possible sulci, we
    select the less used one across regions.
    We assign it the 1st possible region (in the selected smallest ones) (this
    can be improved).
    Then remove the sulcus from other regions, update their sizes.
    And continue until all sulci have been processed.

    Regions sizes (currently) are the number of sulci they contain. This may be
    improved in the future.

    The algorithm tries to assign a sulcus to the smallest region which
    represent it, and avoid "erasing" small regions.
    """
    sulci_regions = {}
    region_sizes = {}
    region_sulci = {}
    for region, sulci_l in regions['brain'].items():
        region_sizes[region] = len(sulci_l)
        region_sulci[region] = set(sulci_l)
        for sulcus in sulci_l:
            sulci_regions.setdefault(sulcus, []).append(region)

    size_reg = {}
    for region, size in region_sizes.items():
        size_reg.setdefault(size, set()).add(region)

    nsulci = len(sulci_regions)
    sulci_region_map = {}
    done = set()
    itera = 0
    # start with sulci from smallest regions in order to avoid erasing them
    while len(done) != nsulci:
        itera += 1
        for size in sorted(size_reg.keys()):
            regionset = size_reg[size]
            # chose less used sulcus in regionset
            sulci = set()
            for region in regionset:
                sulci.update(region_sulci[region])
            sulci.difference_update(done)
            if len(sulci) != 0:
                break
        sizes = [len(sulci_regions[s]) for s in sulci]
        i = np.argmin(sizes)
        sulcus = list(sulci)[i]
        for region in regionset:
            if sulcus in region_sulci[region]:
                break
        done.add(sulcus)
        sulci_region_map[sulcus] = region
        for region2 in sulci_regions[sulcus]:
            if region2 != region:
                sz = region_sizes[region2]
                regionset2 = size_reg[sz]
                regionset2.remove(region2)
                if len(regionset2) == 0:
                    del size_reg[sz]
                if sz > 1:
                    size_reg.setdefault(sz - 1, set()).add(region2)
                    region_sizes[region2] -= 1
                    region_sulci[region2].remove(sulcus)
                else:
                    del region_sizes[region2]
                    del region_sulci[region2]
                    # print('region', region2, 'vanishes.')

    return sulci_region_map


def map_values_in_model(data, sulci_region_map, models, property_name='value',
                        default_value=0.):
    """ Insert values associated with regions into the model graphs.

    Graphs in the models list will be modified.

    Parameters
    ----------
    data: dict
        dict {region: value}, as obtained by read_regions_data()
    sulci_region_map: dict
        sulcus to region assignment map {sulcus: region} as obtained by
        region_to_sulci_mapping()
    models: list of anatomist AGraph objects
        may be one or two graphs (for left, right, both). Can be read using
        load_regions()
    property_name: str
        name of the property in which the values will be stored in model graphs
        vertices.
    default_value: float
        value set for sulci not found in the data dict
    """
    for model in models:
        for v in model.graph().vertices():
            label = v.get('name')
            value = default_value
            if label is not None:
                region = sulci_region_map.get(label)
                if region is not None:
                    value = data.get(region, default_value)
            v[property_name] = value
    for model in models:
        model.setChanged()
        model.notifyObservers()
    a = ana.Anatomist()
    a.execute('GraphDisplayProperties', objects=models,
              display_mode='PropertyMap', display_property=property_name)


def build_glassbrain_palette(name, source_name):
    """ Add a palette in Anatomist, copied from source_name, with the first
    color white and half opacity.
    """
    a = ana.Anatomist()
    pal = a.getPalette(source_name)
    if pal is None:
        raise ValueError(f'palette {source_name} not found')
    npal = a.getPalette(name)
    if npal is None:
        npal = a.createPalette(name)
    cols = np.array(pal.np['v'][:, 0, 0, 0], copy=True)
    cols[0] = [255, 255, 255, 128]
    npal.setColors(cols.ravel(), color_mode='RGBA')


def build_glassbrain_view(regions_csv, column=0, regions_path=None,
                          model_version='2019', hemi='both',
                          palette='glassbrain', bounds=None,
                          zero_centered=False):
    """ Basically prepares everything for a valmues map view.

    Load regions list, regions values data file, sulci model files (graphs),
    set values maps on the models, apply a colormap palette, open an Anatomist
    window, display models in the window.

    Parameters
    ----------
    regions_csv: str or dict or numpy array or list or pandas dataframe
        values assigned to regions to be mapped on the display. Use
        read_regions_data() to read it.
    column: int
        column in a data table (dataframe or CSV) to use for values. see
        read_regions_data() for details.
    regions_path: str
        regions definition (JSON file). Default to Champollion v1 regions.
    model_version: str
        sulci model version to be used. '2008' (SPAM), '2019' (CNN)... The
        sulci list and nomenclature may change with versions.
    hemi: str
        'left', 'right', or 'both'
    palette: str
        Anatomist palette name to be used. Defaults to a slightly custom one
    bounds: list of float
        [min, max] values to be mapped to the palette
    zero_centered: bool
        use True for positive/negative values, and use a zero-centered palette.

    Returns
    -------
    win: AWindow
        Anatomist window opened for the view
    hie: AObject
        Anatomist nomenclature object
    models: list of AGraph
        sulci model graphs
    regions: dict
        regions definition dict
    data: dict
        {region: value} dataset
    """
    data = read_regions_data(regions_csv, column)
    if regions_path is None:
        regions_path = ('/neurospin/dico/data/deep_folding/current/'
                        'sulci_regions_champollion_V1.json')
    if not osp.exists(regions_path):
        print(f'Warning: regions file {regions_path} not found. Falling back '
              'to builtin sulci_regions_overlap.json')
        regions_path = aims.carto.Paths.findResourceFile(
            'nomenclature/translation/sulci_regions_overlap.json')
    regions = load_regions(regions_path)
    hie, model_l, model_r = load_sulci_model(model_version)
    if hemi == 'left':
        models = [model_l]
    elif hemi == 'right':
        models = [model_r]
    else:
        models = [model_l, model_r]

    # sulcus -> region without overlap
    sulci_region_map = region_to_sulci_mapping(regions)

    if palette == 'glassbrain':
        build_glassbrain_palette('glassbrain', 'Yellow-red-fusion')

    map_values_in_model(data, sulci_region_map, models)
    for model in models:
        model.setPalette(palette)
        if bounds:
            model.setPalette(minVal=bounds[0], maxVal=bounds[1],
                             absoluteMode=True, zeroCentered1=zero_centered)

    a = ana.Anatomist()
    win = a.createWindow('3D')
    a.execute('WindowConfig', windows=[win], cursor_visibility=0)
    win.addObjects(models)
    a.execute('SelectByNomenclature', nomenclature=hie,
              names='unknown ventricle_left ventricle_right',
              modifiers='remove')

    return (win, hie, models, regions, data)


def take_glassbrain_snapshot(window, quaternion, size, filename=None):
    """ take a snapshot image from an existing view for a given orientation.

    Orientation is given as a quaternion instead of a direction, because a
    direction does not allow to specify the rotation around the direction axis.
    """
    # # this was for direction instead of quaternion...
    # quat = aims.Quaternion()
    # axis = aims.vectProduct((0., 0., -1.), orientation)
    # angle = np.arcsin(axis.norm())
    # axis.normalize()
    # quat.fromAxis(axis, angle)
    quat = quaternion
    if hasattr(quat, 'vector'):  # aims.Quaternion object
        quat = quat.vector()
    window.camera(view_quaternion=quat)
    window.focusView()

    if size is None:
        w = 0
        h = 0
    else:
        w, h, = size
    image = window.snapshotImage(w, h)
    if filename is not None:
        image.save(filename)

    return image


def default_view_sizes():
    """ returns default view sizes, by just multiplying values from the
    DEFAULT_VIEW_SIZES variable by VIEW_SIZE_FACOR.
    """
    sizes = [[int(s * VIEW_SIZE_FACOR) for s in sz]
             for sz in DEFAULT_VIEW_SIZES]
    return sizes


def glassbrain(regions_csv, filenames, sizes=None, column=0, regions_path=None,
               model_version='2019', hemi='both',
               palette='glassbrain', bounds=None,
               zero_centered=False,
               quaternions=DEFAULT_QUATERNIONS, palette_snapshot=None,
               palette_size=None):
    """ Make a full glassbrain view from a regions values dataset, and take
    several snapshots from it at different orientations

    Parameters
    ----------
    regions_csv: str or dict or numpy array or list or pandas dataframe
        values assigned to regions to be mapped on the display. Use
        read_regions_data() to read it.
    filenames: str or list of str
        output filenames for each view image. If it is a single string, like
        'glassview.jpg', then a number will be appended: 'glassview_01.jpg',
        'glassview_02.jpg' etc. The list is supposed to have the same size as
        the quaternions parameter, otherwise numbers will be appended.
    sizes: list
        list of snapshot sizes [width, height]. If a single size is specified,
        the same is applied for all views. If None, default view sizes will be
        used (see default_view_sizes())
    column: int
        column in a data table (dataframe or CSV) to use for values. see
        read_regions_data() for details.
    regions_path: str
        regions definition (JSON file). Default to Champollion v1 regions.
    model_version: str
        sulci model version to be used. '2008' (SPAM), '2019' (CNN)... The
        sulci list and nomenclature may change with versions.
    hemi: str
        'left', 'right', or 'both'
    palette: str
        Anatomist palette name to be used. Defaults to a slightly custom one
    bounds: list of float
        [min, max] values to be mapped to the palette
    zero_centered: bool
        use True for positive/negative values, and use a zero-centered palette.
    quaternions: list of 4-uple
        list of views orientations. Each element is a quaternion definition: 4
        numbers (x, y, z, w) specifying the rotation axis and sinus of half
        rotation angle. Rotation rotates the camera orientation from the
        default orientation along Z axis (0, 0, 1).
    palette_snapshot: str
        output filename for the palette colormap and scale
    palette_size: list of 2 ints
        size of the palette image snapshot
    """
    res = build_glassbrain_view(regions_csv, column, regions_path,
                                model_version=model_version, hemi=hemi,
                                palette=palette, bounds=bounds,
                                zero_centered=zero_centered)
    window = res[0]
    if sizes is None or len(sizes) == 0:
        sizes = default_view_sizes()
    palette_snapshot_done = False

    for i, quaternion in enumerate(quaternions):
        if filenames is not None and isinstance(filenames, (list, tuple)) \
                and len(filenames) > i:
            filename = filenames[i]
        elif len(filenames) != 0:
            if isinstance(filenames, (list, tuple)):
                filenames = filenames[-1]
            dirname = osp.dirname(filenames)
            bname = osp.basename(filenames)
            lfilename = bname.rsplit('.', 1)
            ext = '.'.join([''] + lfilename[1:])
            nfilename = f'{lfilename[0]}_{i:0>2}{ext}'
            filename = osp.join(dirname, nfilename)
        else:
            filename = None
        if not hasattr(sizes[0], '__iter__'):
            size = sizes
        elif len(sizes) > i:
            size = sizes[i]
        else:
            size = sizes[-1]
        print('filename:', filename, ', size:', size)
        take_glassbrain_snapshot(window, quaternion, size, filename)

        if not palette_snapshot_done:
            palette_snapshot_done = True
            if palette_snapshot is not None:
                import paletteViewer

                models = res[2]
                paletteViewer.savePaletteImage(models[0], palette_snapshot,
                                               palette_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'build glassbrain-like snapshots for a set of data values (scalars) '
        'associated with regions')
    parser.add_argument('-i', '--input',
                        help='regions data values. May be a JSON dict '
                        '{region: value}, or a CSV table.')
    parser.add_argument('-o', '--output', nargs='+',
                        help='output snapshot image(s). Several -o options '
                        'allowed. If a single one is used, a number will be '
                        'appended for each image.')
    parser.add_argument('-s', '--size', nargs='*',
                        help='snapshot images sizes (as a json/python list: '
                        '[800, 600], several -s options allowed)')
    parser.add_argument('-c', '--column', type=int, default=0,
                        help='table column to take values from in the data '
                        'table, if the .CSV file contains several columns.')
    parser.add_argument('-r', '--regions',
                        help='regions definition (JSON file). Default to '
                        'Champollion v1 regions.')
    parser.add_argument('-v', '--version', default='2019',
                        help='sulci model version to be used. "2008" (SPAM), '
                        '"2019" (CNN)... The sulci list and nomenclature may '
                        'change with versions. Defauilt: 2019')
    parser.add_argument('--side', default='both', nargs='*',
                        help='hemisphere to display: "left", "right", or '
                        '"both". Default: both')
    parser.add_argument('-p', '--palette', default='glassbrain',
                        help='Anatomist palette name to be used. Defaults to '
                        'a slightly custom one.')
    parser.add_argument('--min', type=float,
                        help='min value to be mapped to the palette')
    parser.add_argument('--max', type=float,
                        help='max value to be mapped to the palette')
    parser.add_argument('-z', '--zero_centered', action='store_true',
                        help='use it for positive/negative values, and use a '
                        'zero-centered palette.')
    parser.add_argument('-q', '--quaternions', nargs='*',
                        help='views orientations. Several -q options may be '
                        'used, each to specivy a snapshot viuew orientation. '
                        'Each is a quaternion definition: 4 numbers '
                        '[x, y, z, w] (in JSON format) specifying the '
                        'rotation axis and sinus of half rotation angle. '
                        'Rotation rotates the camera orientation from the '
                        'default orientation along Z axis (0, 0, 1). Defaults '
                        'to 6 directions from the 6 faces of a cube.')
    parser.add_argument('-op', '--output_palette',
                        help='save palette colormap and scale image')
    parser.add_argument('-ps', '--palette_size',
                        help='output palette image size, JSON list of 2 ints')
    options = parser.parse_args()
    regions_csv = options.input
    filenames = options.output
    if options.size is None:
        sizes = None
    else:
        sizes = [json.loads(x) for x in options.size]
    column = options.column
    regions_path = options.regions
    model_version = options.version
    hemi = options.side
    palette = options.palette
    bounds = None
    if options.min is not None or options.max is not None:
        bounds = [options.min, options.max]
    zero_centered = options.zero_centered
    quaternions = DEFAULT_QUATERNIONS
    if options.quaternions:
        quaternions = [json.loads(x) for x in options.quaternions]
    palette_snapshot = options.output_palette
    palette_size = None
    if options.palette_size is not None:
        palette_size = json.loads(options.palette_size)

    print('regions_csv:', regions_csv)
    print('filenames:', filenames)
    print('sizes:', sizes)
    print('column:', column)
    print('regions_path:', regions_path)
    print('model_version:', model_version)
    print('hemi:', hemi)
    print('palette:', palette)
    print('bounds:', bounds)
    print('zero_centered:', zero_centered)
    print('quaternions:', quaternions)
    print('palette_snapshot:', palette_snapshot)

    if regions_csv is None:
        raise ValueError('Input regions data is required.')
    if filenames is None:
        raise ValueError('an output filename is required.')

    import anatomist.headless as ana

    a = ana.Anatomist()

    glassbrain(regions_csv, filenames, sizes=sizes, column=column,
               regions_path=regions_path, model_version=model_version,
               hemi=hemi, palette=palette, bounds=bounds,
               zero_centered=zero_centered, quaternions=quaternions,
               palette_snapshot=palette_snapshot, palette_size=palette_size)
