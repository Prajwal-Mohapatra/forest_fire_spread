
// =================== CONFIG ===================
var uttarakhand = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
  .filter(ee.Filter.eq('ADM1_NAME', 'Uttarakhand'));

// Load SRTM DEM (30m)
var dem = ee.Image("USGS/SRTMGL1_003").clip(uttarakhand);

// Optional: derive slope & aspect
var terrain = ee.Terrain.products(dem);
var slope = terrain.select('slope');
var aspect = terrain.select('aspect');

// =================== EXPORT ===================
// Export DEM
Export.image.toDrive({
  image: dem,
  description: 'Uttarakhand_DEM_SRTM_30m',
  folder: 'Dataset_PS1',
  fileNamePrefix: 'uttarakhand_dem_2016',
  region: uttarakhand.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Optional: Export Slope
Export.image.toDrive({
  image: slope,
  description: 'Uttarakhand_Slope_SRTM_30m',
  folder: 'Dataset_PS1',
  fileNamePrefix: 'uttarakhand_slope_2016',
  region: uttarakhand.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// Optional: Export Aspect
Export.image.toDrive({
  image: aspect,
  description: 'Uttarakhand_Aspect_SRTM_30m',
  folder: 'Dataset_PS1',
  fileNamePrefix: 'uttarakhand_aspect_2016',
  region: uttarakhand.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// =================== VISUALIZATION ===================
Map.centerObject(uttarakhand, 8);
Map.addLayer(dem, {min: 0, max: 4000, palette: ['white', 'gray', 'black']}, 'DEM');
Map.addLayer(slope, {min: 0, max: 60, palette: ['white', 'orange', 'red']}, 'Slope');
Map.addLayer(aspect, {min: 0, max: 360, palette: ['blue', 'green', 'yellow']}, 'Aspect');
