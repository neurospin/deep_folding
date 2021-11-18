from soma import aims

vol = aims.Volume((260,311,260), dtype='S16')

graph = aims.read(graph_filename)

for vertex in graph.vertices():
    for bucket_name, value in {'aims_other':100, 'aims_ss':60, 'aims_bottom':30}.items():
        bucket = vertex.get(bucket_name)
            if bucket is not None:
                voxels_real = np.asarray(
                    [np.array(voxel)) for voxel in bucket[0].keys()])