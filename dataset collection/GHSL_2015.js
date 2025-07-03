// Config
var uttarakhand = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
  .filter(ee.Filter.eq('ADM1_NAME', 'Uttarakhand'));

// Load built-up surface for 2015
var ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2015')
  .select('built_surface')
  .clip(uttarakhand)
  .rename('ghsl_built_2015');

// Binary urban mask (optional threshold)
var ghsl_bin = ghsl.gt(1000).rename('ghsl_urban');  // ≥1000 m² built-up

// Export
Export.image.toDrive({
  image: ghsl,
  description: 'Uttarakhand_GHSL_Built2015',
  folder: 'Dataset_PS1',
  fileNamePrefix: 'ghsl_uttarakhand_built2015',
  region: uttarakhand.geometry(),
  scale: 30, //resampled to 30m
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: ghsl_bin,
  description: 'Uttarakhand_GHSL_UrbanMask2015',
  folder: 'Dataset_PS1',
  fileNamePrefix: 'ghsl_uttarakhand_urbanmask2015',
  region: uttarakhand.geometry(),
  scale: 30, //resampled to 30m
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Visualize
Map.centerObject(uttarakhand, 8);
Map.addLayer(ghsl, {min:0, max:5000, palette:['white','orange']}, 'GHSL built-up [m²]');
Map.addLayer(ghsl_bin, {min:0, max:1, palette:['white','black']}, 'Urban mask');
