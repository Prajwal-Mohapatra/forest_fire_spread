// =================== CONFIGURATION ===================
var uttarakhand = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
  .filter(ee.Filter.eq('ADM1_NAME', 'Uttarakhand'));

// Load ESA WorldCover 2020 (10m resolution)
var lulc = ee.Image("ESA/WorldCover/v100/2020")
  .select('Map')
  .clip(uttarakhand);

// =================== RECLASSIFY TO FUEL CLASSES ===================
// Reference: https://esa-worldcover.org/en/legend
// Mapping ESA classes to fuel load: 0 = No fuel, 3 = High fuel

var fuel_map = lulc.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], // ESA class codes
  [3, 1, 2, 1, 0, 0, 0, 2, 0, 3, 2]              // Your custom fuel classes
).rename('fuel_class');

// =================== EXPORT ===================
Export.image.toDrive({
  image: fuel_map,
  description: 'Uttarakhand_FuelMap_ESAWC2020',
  folder: 'Dataset_PS1',
  fileNamePrefix: 'uttarakhand_fuel_map_2020',
  region: uttarakhand.geometry(),
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// =================== VISUALIZATION ===================
Map.centerObject(uttarakhand, 8);
Map.addLayer(fuel_map, {min: 0, max: 3, palette: ['gray', 'yellow', 'orange', 'darkgreen']}, 'Fuel Mask (0–3)');
